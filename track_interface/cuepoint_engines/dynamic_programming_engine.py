from track_interface.cuepoint_engines.heuristics import (
    FirstBeatsOnly,
    RestrictedMeasureIncrements,
    SongEndCuepoint,
    SongStartCuepoint,
)
from track_interface.cuepoint_engines.cuepoint_engine import BeatGrid
from track_interface.cuepoint_engines.changepoint_engine import ChangePointEngine
from track_interface.cuepoint_engines.cache import (
    CACHE_ENABLED,
    convert_to_key,
    get,
    put,
)
import bisect
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import librosa
import math
import statistics
from collections import defaultdict
from typing import List, Literal, Any, Tuple, Optional
from dataclasses import dataclass
import os
import io
import base64
from scipy.signal import find_peaks
from scipy.sparse import diags
import sklearn.metrics.pairwise as pairwise_distances
import warnings


@dataclass(frozen=True)
class DynamicProgrammingEngineParams:
    frames_per_measure: int = 96
    similarity_func: str = "cosine"
    penalty_weight: float = 1.0
    penalty_func: str = "modulo8"
    min_size: int = 1
    max_size: int = 32
    bands_number: Optional[int] = 7
    n_mfcc: int = 13
    hop_length: int = 512
    sample_rate: int = 22050


class DynamicProgrammingEngine(ChangePointEngine):
    """
    Implementation of Correlation Block-Matching (CBM) algorithm from https://arxiv.org/pdf/2210.15356.

    This algorithm uses dynamic programming to segment music based on barwise autosimilarity,
    focusing on the criteria of homogeneity to estimate segments.

    Reference:
    A. Marmoret, J. E. Cohen, F. Bimbot. Barwise Music Structure Analysis with the
    Correlation Block Matching Segmentation Algorithm. Transactions of the International
    Society for Music Information Retrieval, 6(1), 167--185.
    """

    def __init__(self, params: Optional[DynamicProgrammingEngineParams] = None):
        if params:
            self.params = params
        else:
            self.params = DynamicProgrammingEngineParams()

    # ========== Autosimilarity Computation Methods ==========

    def l2_normalize_barwise(self, an_array: np.ndarray) -> np.ndarray:
        """
        Normalizes the array barwise (i.e., in its first dimension) by the l_2 norm.
        Null values are replaced by the small positive value of 10^{-10}.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in true_divide"
            )
            an_array_T = an_array.T / np.linalg.norm(an_array, axis=1)
            an_array_T = np.where(np.isnan(an_array_T), 1e-10, an_array_T)
        return an_array_T.T

    def get_cosine_autosimilarity(self, an_array: np.ndarray) -> np.ndarray:
        """
        Computes the autosimilarity matrix with the cosine similarity function.
        The cosine similarity is the normalized dot product between two bars.
        """
        if isinstance(an_array, list):
            this_array = np.array(an_array)
        else:
            this_array = an_array
        this_array = self.l2_normalize_barwise(this_array)
        return this_array @ this_array.T

    def get_autocorrelation_autosimilarity(self, an_array: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Computes the autosimilarity matrix with autocorrelation (covariance) similarity.
        This corresponds to the dot product of centered features.
        """
        if isinstance(an_array, list):
            this_array = np.array(an_array)
        else:
            this_array = an_array
        this_array = this_array - this_array.mean(axis=0)
        if normalize:
            this_array = self.l2_normalize_barwise(this_array)
        return this_array @ this_array.T

    def get_gamma_std(self, an_array: np.ndarray, scaling_factor: float = 1,
                      no_diag: bool = True, normalize: bool = True) -> float:
        """
        Computes default gamma value for RBF similarity as a function of standard deviation.
        """
        if normalize:
            an_array = self.l2_normalize_barwise(an_array)
        euc_dist = pairwise_distances.euclidean_distances(an_array)
        if not no_diag:
            return scaling_factor / (2 * np.std(euc_dist))
        else:
            for i in range(len(euc_dist)):
                euc_dist[i, i] = float("NaN")
            return scaling_factor / (2 * np.nanstd(euc_dist))

    def get_rbf_autosimilarity(self, an_array: np.ndarray, gamma: Optional[float] = None,
                               normalize: bool = True) -> np.ndarray:
        """
        Computes the autosimilarity matrix using Radial Basis Function (RBF).
        The RBF is the exponent of the negative euclidean distance between features.
        """
        if isinstance(an_array, list):
            this_array = np.array(an_array)
        else:
            this_array = an_array
        if gamma is None:
            gamma = self.get_gamma_std(this_array, scaling_factor=1, no_diag=True, normalize=normalize)
        if normalize:
            this_array = self.l2_normalize_barwise(this_array)
        return pairwise_distances.rbf_kernel(this_array, gamma=gamma)

    def compute_autosimilarity(self, an_array: np.ndarray, similarity_type: str = "cosine",
                               gamma: Optional[float] = None, normalize: bool = True) -> np.ndarray:
        """
        High-level function to compute autosimilarity with different similarity functions.
        """
        if similarity_type.lower() == "cosine":
            return self.get_cosine_autosimilarity(an_array)
        elif similarity_type.lower() in ["covariance", "autocorrelation"]:
            return self.get_autocorrelation_autosimilarity(an_array, normalize=normalize)
        elif similarity_type.lower() == "rbf":
            return self.get_rbf_autosimilarity(an_array, gamma, normalize=normalize)
        else:
            raise ValueError(
                f"Incorrect similarity type: {similarity_type}. Should be cosine, covariance or rbf."
            )

    # ========== CBM Algorithm Core Methods ==========

    def compute_all_kernels(self, max_size: int, bands_number: Optional[int] = None) -> List[np.ndarray]:
        """
        Precompute kernels from size 0 to max_size for acceleration.
        Kernels are square matrices used for correlation computation.
        """
        kernels = [[0]]
        for p in range(1, max_size + 1):
            if bands_number is None or p < bands_number:
                kern = np.ones((p, p)) - np.identity(p)
            else:
                k = np.array(
                    [
                        np.ones(p - i)
                        for i in np.abs(range(-bands_number, bands_number + 1))
                    ],
                    dtype=object,
                )
                offset = [i for i in range(-bands_number, bands_number + 1)]
                kern = diags(k, offset).toarray() - np.identity(p)
            kernels.append(kern)
        return kernels

    def correlation_cost(self, cropped_autosimilarity: np.ndarray, kernels: List[np.ndarray]) -> float:
        """
        Calculates the correlation/convolution measure on a segment of the autosimilarity matrix.
        Normalizes result by segment size squared.
        """
        p = len(cropped_autosimilarity)
        kern = kernels[p]
        return np.sum(np.multiply(kern, cropped_autosimilarity)) / p**2

    def correlation_entire_matrix_computation(self, autosimilarity_array: np.ndarray,
                                              kernels: List[np.ndarray], kernel_size: int = 8) -> np.ndarray:
        """
        Applies correlation measures across the complete autosimilarity matrix
        using a fixed kernel size.
        """
        cost = np.zeros(len(autosimilarity_array))
        for i in range(kernel_size, len(autosimilarity_array)):
            cost[i] = self.correlation_cost(
                autosimilarity_array[i - kernel_size: i, i - kernel_size: i], kernels
            )
        return cost

    def penalty_cost(self, segment_length: int, penalty_func: str = "modulo8") -> float:
        """
        Applies penalty based on segment length to enforce domain knowledge about
        typical segment lengths (e.g., 4-bar, 8-bar sections in music).
        """
        if penalty_func == "modulo8":
            if segment_length == 8:
                return 0
            elif segment_length % 4 == 0:
                return 1 / 4
            elif segment_length % 2 == 0:
                return 1 / 2
            else:
                return 1
        if penalty_func == "modulo4":
            if segment_length % 4 == 0:
                return 0
            elif segment_length % 2 == 0:
                return 1 / 2
            else:
                return 1
        if penalty_func == "modulo8modulo4":
            if segment_length == 8:
                return 0
            elif segment_length == 4:
                return 1 / 4
            elif segment_length % 2 == 0:
                return 1 / 2
            else:
                return 1
        if penalty_func == "target_deviation_8_alpha_half":
            return abs(segment_length - 8) ** (1 / 2)
        if penalty_func == "target_deviation_8_alpha_one":
            return abs(segment_length - 8)
        if penalty_func == "target_deviation_8_alpha_two":
            return abs(segment_length - 8) ** 2
        else:
            raise ValueError(f"Penalty function not understood: {penalty_func}.")

    def possible_segment_start(self, idx: int, min_size: int = 1,
                               max_size: Optional[int] = None) -> range:
        """
        Generates all valid starting positions for segments given an endpoint,
        respecting minimum and maximum size constraints.
        """
        if min_size < 1:
            raise ValueError(
                f"Invalid minimal size: {min_size} (No segment should be 0 or negative size)."
            )
        if max_size is None:
            return range(0, idx - min_size + 1)
        else:
            if idx >= max_size:
                return range(idx - max_size, idx - min_size + 1)
            elif idx >= min_size:
                return range(0, idx - min_size + 1)
            else:
                return []

    def compute_cbm(self, autosimilarity: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
        """
        Main dynamic programming function for music segmentation.
        Maximizes overall score across segments using correlation-based block matching.

        Returns:
            segments: List of (start_idx, end_idx) tuples for each segment
            best_score: The optimal score achieved
        """
        min_size = self.params.min_size
        max_size = self.params.max_size
        penalty_weight = self.params.penalty_weight
        penalty_func = self.params.penalty_func
        bands_number = self.params.bands_number

        scores = [-math.inf for i in range(len(autosimilarity))]
        segments_best_starts = [None for i in range(len(autosimilarity))]
        segments_best_starts[0] = 0
        scores[0] = 0

        kernels = self.compute_all_kernels(max_size, bands_number=bands_number)
        max_conv_eight = np.amax(
            self.correlation_entire_matrix_computation(autosimilarity, kernels)
        )

        for current_idx in range(1, len(autosimilarity)):
            for possible_start_idx in self.possible_segment_start(
                current_idx, min_size=min_size, max_size=max_size
            ):
                if possible_start_idx < 0:
                    raise ValueError(
                        f"Invalid value of start index: {possible_start_idx}, shouldn't happen."
                    )

                conv_cost = self.correlation_cost(
                    autosimilarity[
                        possible_start_idx:current_idx, possible_start_idx:current_idx
                    ],
                    kernels,
                )

                segment_length = current_idx - possible_start_idx
                penalty = self.penalty_cost(segment_length, penalty_func)

                this_segment_cost = (
                    conv_cost * segment_length
                    - penalty * penalty_weight * max_conv_eight
                )

                if possible_start_idx == 0:
                    if this_segment_cost > scores[current_idx]:
                        scores[current_idx] = this_segment_cost
                        segments_best_starts[current_idx] = 0
                else:
                    if (
                        scores[possible_start_idx] + this_segment_cost > scores[current_idx]
                    ):
                        scores[current_idx] = scores[possible_start_idx] + this_segment_cost
                        segments_best_starts[current_idx] = possible_start_idx

        # Backtrack to find optimal segmentation
        segments = [
            (segments_best_starts[len(autosimilarity) - 1], len(autosimilarity) - 1)
        ]
        precedent_frontier = segments_best_starts[len(autosimilarity) - 1]
        while precedent_frontier > 0:
            segments.append((segments_best_starts[precedent_frontier], precedent_frontier))
            precedent_frontier = segments_best_starts[precedent_frontier]
            if precedent_frontier is None:
                raise ValueError(
                    "Dynamic programming algorithm took an impossible path."
                )
        return segments[::-1], scores[-1]

    # ========== Legacy/Compatibility Methods ==========

    def regularize(self, num_measures: int) -> float:
        """Legacy method - penalty is now handled by penalty_cost()."""
        return self.penalty_cost(num_measures, self.params.penalty_func)

    def calculate_num_frames(self, beat_grid: List[int]) -> int:
        """Calculate number of frames based on beat grid."""
        return self.params.frames_per_measure // 4 * len(beat_grid)

    # ========== Main Interface Method ==========

    def generate_cuepoints(self, file_path: str, beat_grid: BeatGrid) -> List[int]:
        """
        Generate cuepoints for a track using the CBM algorithm.

        Args:
            file_path: Path to the audio file
            beat_grid: Beat grid information for the track

        Returns:
            List of cuepoint indices corresponding to beat positions
        """
        # Load audio and extract MFCC features
        y, sr = librosa.load(file_path, sr=self.params.sample_rate)
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.params.n_mfcc,
            hop_length=self.params.hop_length
        )

        # Transpose to get (time, features) shape
        mfcc = mfcc.T

        # Compute autosimilarity matrix
        autosimilarity = self.compute_autosimilarity(
            mfcc,
            similarity_type=self.params.similarity_func
        )

        # Run CBM algorithm to get segments
        segments, score = self.compute_cbm(autosimilarity)

        # Convert segment boundaries to beat grid indices
        # Each segment boundary represents a potential cuepoint
        cuepoint_indices = []

        # Map MFCC frame indices to beat indices
        hop_length = self.params.hop_length
        sr = self.params.sample_rate

        for start_idx, end_idx in segments:
            # Convert frame index to time
            time_in_seconds = librosa.frames_to_time(
                start_idx,
                sr=sr,
                hop_length=hop_length
            )
            time_in_ms = int(time_in_seconds * 1000)

            # Find closest beat in beat grid
            print(beat_grid, time_in_ms)
            beat_idx = bisect.bisect_left(beat_grid, time_in_ms)
            if beat_idx < len(beat_grid):
                cuepoint_indices.append(beat_idx)

        # Remove duplicates and sort
        cuepoint_indices = sorted(list(set(cuepoint_indices)))

        return cuepoint_indices
