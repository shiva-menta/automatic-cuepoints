"""
Correlation Block-Matching (CBM) Engine for music structure segmentation.

Based on the paper:
A. Marmoret, J. E. Cohen, F. Bimbot. "Barwise Music Structure Analysis with the
Correlation Block Matching Segmentation Algorithm." Transactions of the International
Society for Music Information Retrieval, 6(1), 167--185, 2023.
DOI: https://doi.org/10.5334/tismir.167

Implementation adapted from MSAF (https://github.com/urinieto/msaf).
"""

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import librosa
import numpy as np
import sklearn.metrics.pairwise as pairwise_distances
from scipy.sparse import diags

from track_interface.cuepoint_engines.cuepoint_engine import CuepointEngine
from track_interface.cuepoint_engines.heuristics import (
    DPMeasureAlignment,
    FirstBeatsOnly,
    SongEndCuepoint,
    SongStartCuepoint,
)
from track_interface.types import BeatGrid, Cuepoint, CuepointList


@dataclass(frozen=True)
class CBMEngineParams:
    # Feature extraction
    n_mels: int = 80
    hop_length: int = 512
    n_fft: int = 2048

    # Self-similarity computation
    ssm_type: str = "rbf"  # "cosine", "autocorrelation", or "rbf"

    # CBM algorithm parameters
    min_size: int = 4  # Minimum 4 bars per segment
    max_size: int = 32
    penalty_weight: float = 0.15  # Original penalty weight
    penalty_func: str = "modulo8"  # Original 8-bar preference
    bands_number: Optional[int] = 7  # None for full kernel, 7 for 7-band kernel

    # Post-processing parameters
    novelty_threshold: float = 0.05  # Minimum novelty to keep a boundary

    debug_mode: bool = False


class CBMEngine(CuepointEngine):
    """
    CBM (Correlation Block-Matching) segmentation engine.

    Uses dynamic programming to find optimal segmentation maximizing
    segment homogeneity scores on a barwise self-similarity matrix.
    """

    params: CBMEngineParams

    def _get_default_params(self) -> CBMEngineParams:
        return CBMEngineParams()

    def _get_first_beat_timestamps(self, beat_grid: BeatGrid) -> List[int]:
        """Get timestamps of first beats (start of each bar)."""
        return [beat_tuple[2] for beat_tuple in beat_grid if beat_tuple[0] == 1]

    def _compute_barwise_features(
        self, file_path: str, beat_grid: BeatGrid
    ) -> np.ndarray:
        """
        Compute barwise mel spectrogram features.

        Returns:
            np.ndarray of shape (n_bars, n_features) where n_features = n_mels * frames_per_bar
        """
        y, sr = librosa.load(file_path, sr=None)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.params.n_mels,
            hop_length=self.params.hop_length,
            n_fft=self.params.n_fft,
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Get bar boundaries
        first_beat_timestamps = self._get_first_beat_timestamps(beat_grid)
        if len(first_beat_timestamps) < 2:
            raise ValueError("Need at least 2 bars for CBM segmentation")

        # Convert timestamps to frames
        bar_frames = librosa.time_to_frames(
            [ts / 1000.0 for ts in first_beat_timestamps],
            sr=sr,
            hop_length=self.params.hop_length,
        )

        # Aggregate features per bar
        barwise_features = []
        for i in range(len(bar_frames) - 1):
            start_frame = bar_frames[i]
            end_frame = bar_frames[i + 1]

            if end_frame > log_mel.shape[1]:
                end_frame = log_mel.shape[1]
            if start_frame >= end_frame:
                continue

            # Take mean across time within each bar
            bar_feature = np.mean(log_mel[:, start_frame:end_frame], axis=1)
            barwise_features.append(bar_feature)

        # Handle last bar
        if bar_frames[-1] < log_mel.shape[1]:
            bar_feature = np.mean(log_mel[:, bar_frames[-1] :], axis=1)
            barwise_features.append(bar_feature)

        return np.array(barwise_features)

    def _l2_normalize_barwise(self, an_array: np.ndarray) -> np.ndarray:
        """Normalize the array barwise by the L2 norm."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in true_divide"
            )
            norms = np.linalg.norm(an_array, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-10, norms)
            return an_array / norms

    def _get_gamma_std(
        self, an_array: np.ndarray, normalize: bool = True
    ) -> float:
        """Compute default gamma for RBF based on pairwise distance distribution."""
        if normalize:
            an_array = self._l2_normalize_barwise(an_array)
        euc_dist = pairwise_distances.euclidean_distances(an_array)
        # Exclude diagonal
        np.fill_diagonal(euc_dist, np.nan)
        return 1.0 / (2 * np.nanstd(euc_dist))

    def _compute_self_similarity(self, features: np.ndarray) -> np.ndarray:
        """
        Compute self-similarity matrix using the specified similarity function.
        """
        ssm_type = self.params.ssm_type.lower()

        if ssm_type == "cosine":
            normalized = self._l2_normalize_barwise(features)
            return normalized @ normalized.T

        elif ssm_type in ("covariance", "autocorrelation"):
            centered = features - features.mean(axis=0)
            normalized = self._l2_normalize_barwise(centered)
            return normalized @ normalized.T

        elif ssm_type == "rbf":
            gamma = self._get_gamma_std(features, normalize=True)
            normalized = self._l2_normalize_barwise(features)
            return pairwise_distances.rbf_kernel(normalized, gamma=gamma)

        else:
            raise ValueError(
                f"Unknown SSM type: {ssm_type}. Use 'cosine', 'autocorrelation', or 'rbf'"
            )

    def _compute_all_kernels(self, max_size: int) -> List[np.ndarray]:
        """
        Precompute all kernels of size 0 to max_size.

        The kernel emphasizes similarity within local bands around the diagonal,
        excluding the main diagonal itself.
        """
        kernels = [np.array([0])]  # Size 0 placeholder
        bands_number = self.params.bands_number

        for p in range(1, max_size + 1):
            if bands_number is None or p < bands_number:
                # Full kernel: all off-diagonal elements
                kern = np.ones((p, p)) - np.identity(p)
            else:
                # Band kernel: only elements within bands_number of the diagonal
                k = np.array(
                    [np.ones(p - abs(i)) for i in range(-bands_number, bands_number + 1)],
                    dtype=object,
                )
                offset = list(range(-bands_number, bands_number + 1))
                kern = diags(k, offset).toarray() - np.identity(p)
            kernels.append(kern)

        return kernels

    def _correlation_cost(
        self, cropped_ssm: np.ndarray, kernels: List[np.ndarray]
    ) -> float:
        """Compute correlation/convolution measure on a segment of the SSM."""
        p = len(cropped_ssm)
        if p == 0:
            return 0.0
        kern = kernels[p]
        return np.sum(np.multiply(kern, cropped_ssm)) / (p**2)

    def _correlation_entire_matrix(
        self, ssm: np.ndarray, kernels: List[np.ndarray], kernel_size: int = 8
    ) -> np.ndarray:
        """Compute correlation measure across the entire SSM with fixed kernel size."""
        cost = np.zeros(len(ssm))
        for i in range(kernel_size, len(ssm)):
            cost[i] = self._correlation_cost(
                ssm[i - kernel_size : i, i - kernel_size : i], kernels
            )
        return cost

    def _penalty_cost(self, segment_length: int) -> float:
        """
        Compute penalty cost based on segment length.

        Penalizes segments that don't conform to expected musical structure
        (8-bar, 4-bar, or even-length segments).
        """
        penalty_func = self.params.penalty_func

        if penalty_func == "modulo8":
            if segment_length == 8:
                return 0.0
            elif segment_length % 4 == 0:
                return 0.25
            elif segment_length % 2 == 0:
                return 0.5
            else:
                return 1.0

        elif penalty_func == "modulo16":
            # Prefer 16-bar segments, then 8-bar, then multiples of 4
            if segment_length == 16:
                return 0.0
            elif segment_length == 8 or segment_length == 32:
                return 0.15
            elif segment_length % 8 == 0:
                return 0.25
            elif segment_length % 4 == 0:
                return 0.5
            elif segment_length % 2 == 0:
                return 0.75
            else:
                return 1.0

        elif penalty_func == "modulo4":
            if segment_length % 4 == 0:
                return 0.0
            elif segment_length % 2 == 0:
                return 0.5
            else:
                return 1.0

        elif penalty_func == "modulo8modulo4":
            if segment_length == 8:
                return 0.0
            elif segment_length == 4:
                return 0.25
            elif segment_length % 2 == 0:
                return 0.5
            else:
                return 1.0

        elif penalty_func == "target_deviation_8_alpha_half":
            return abs(segment_length - 8) ** 0.5

        elif penalty_func == "target_deviation_8_alpha_one":
            return abs(segment_length - 8)

        elif penalty_func == "target_deviation_8_alpha_two":
            return abs(segment_length - 8) ** 2

        else:
            raise ValueError(f"Unknown penalty function: {penalty_func}")

    def _possible_segment_starts(self, idx: int) -> List[int]:
        """Generate all possible segment start indices for a given end index."""
        min_size = self.params.min_size
        max_size = self.params.max_size

        if min_size < 1:
            raise ValueError(f"Invalid min_size: {min_size}")

        if idx >= max_size:
            return list(range(idx - max_size, idx - min_size + 1))
        elif idx >= min_size:
            return list(range(0, idx - min_size + 1))
        else:
            return []

    def _compute_cbm(
        self, ssm: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], float]:
        """
        Dynamic programming CBM algorithm.

        Finds optimal segmentation maximizing total score (homogeneity - penalty).

        Returns:
            Tuple of (segments as list of (start, end) tuples, final score)
            Segments are (start_bar_inclusive, end_bar_exclusive).
        """
        n = len(ssm)
        min_size = self.params.min_size
        max_size = self.params.max_size
        penalty_weight = self.params.penalty_weight

        # DP arrays - index i represents "optimal segmentation for bars [0, i)"
        # We need n+1 elements to represent "all n bars segmented"
        scores = [-math.inf] * (n + 1)
        best_starts = [None] * (n + 1)
        scores[0] = 0  # Empty prefix has score 0

        # Precompute kernels and normalization factor
        kernels = self._compute_all_kernels(max_size)
        max_conv_eight = np.amax(
            self._correlation_entire_matrix(ssm, kernels, kernel_size=8)
        )

        # Handle edge case where all correlations are near zero
        if max_conv_eight < 1e-10:
            max_conv_eight = 1.0

        # DP: find optimal segmentation
        # current_idx represents "end of segmentation so far" (exclusive)
        for current_idx in range(min_size, n + 1):
            for start_idx in self._possible_segment_starts(current_idx):
                if start_idx < 0:
                    continue

                # Compute segment score for bars [start_idx, current_idx)
                segment_ssm = ssm[start_idx:current_idx, start_idx:current_idx]
                conv_cost = self._correlation_cost(segment_ssm, kernels)
                segment_length = current_idx - start_idx
                penalty_cost = self._penalty_cost(segment_length)

                # Total segment score
                segment_score = (
                    conv_cost * segment_length
                    - penalty_cost * penalty_weight * max_conv_eight
                )

                # Update if this is the best path to current_idx
                total_score = scores[start_idx] + segment_score
                if total_score > scores[current_idx]:
                    scores[current_idx] = total_score
                    best_starts[current_idx] = start_idx

        # Backtrack to find segments
        # Start from the full segmentation (all n bars)
        segments = []
        current = n
        while current > 0 and best_starts[current] is not None:
            start = best_starts[current]
            segments.append((start, current))
            current = start

        return segments[::-1], scores[n]

    def _filter_by_novelty(
        self,
        segments: List[Tuple[int, int]],
        ssm: np.ndarray,
        threshold: float = 0.15,
    ) -> List[Tuple[int, int]]:
        """
        Filter segments by keeping only boundaries with significant novelty.

        Novelty is measured as the difference between within-segment similarity
        and cross-segment similarity. High novelty = segments are very different.

        Args:
            segments: List of (start, end) tuples
            ssm: Self-similarity matrix
            threshold: Minimum novelty score to keep a boundary (0-1 scale)

        Returns:
            Filtered list of segments with low-novelty boundaries merged
        """
        if len(segments) <= 1:
            return segments

        merged = [segments[0]]

        for i in range(1, len(segments)):
            prev_start, prev_end = merged[-1]
            curr_start, curr_end = segments[i]

            # Compute within-segment similarities
            prev_within = np.mean(ssm[prev_start:prev_end, prev_start:prev_end])
            curr_within = np.mean(ssm[curr_start:curr_end, curr_start:curr_end])
            avg_within = (prev_within + curr_within) / 2

            # Compute cross-segment similarity
            cross_sim = np.mean(ssm[prev_start:prev_end, curr_start:curr_end])

            # Novelty = how much less similar are the segments to each other
            # compared to their internal similarity
            novelty = avg_within - cross_sim

            if novelty >= threshold:
                # Keep boundary - segments are different enough
                merged.append(segments[i])
            else:
                # Merge segments - they're too similar
                merged[-1] = (prev_start, curr_end)

        return merged

    def generate_cuepoints(self, file_path: str, beat_grid: BeatGrid) -> CuepointList:
        """
        Generate cuepoints using CBM segmentation.

        Returns:
            List of Cuepoint objects with timestamps at segment boundaries.
        """
        first_beat_timestamps = self._get_first_beat_timestamps(beat_grid)

        if len(first_beat_timestamps) < 3:
            # Not enough bars for meaningful segmentation
            return [Cuepoint(timestamp=first_beat_timestamps[0], label="")]

        # Compute barwise features and self-similarity matrix
        features = self._compute_barwise_features(file_path, beat_grid)

        if len(features) < 2:
            return [Cuepoint(timestamp=first_beat_timestamps[0], label="")]

        ssm = self._compute_self_similarity(features)

        if self.params.debug_mode:
            print(f"CBM: {len(features)} bars, SSM shape: {ssm.shape}")
            print(f"CBM params: min_size={self.params.min_size}, max_size={self.params.max_size}, penalty_weight={self.params.penalty_weight}")

        # Run CBM segmentation
        segments, score = self._compute_cbm(ssm)

        if self.params.debug_mode:
            print(f"CBM raw segments (bar indices): {segments}")
            segment_lengths = [end - start for start, end in segments]
            print(f"CBM segment lengths: {segment_lengths}")
            print(f"CBM score: {score}")

        # Filter by novelty - remove low-confidence boundaries
        if self.params.novelty_threshold > 0:
            segments = self._filter_by_novelty(
                segments, ssm, self.params.novelty_threshold
            )
            if self.params.debug_mode:
                print(f"CBM after novelty filter: {segments}")
                novelty_lengths = [end - start for start, end in segments]
                print(f"CBM post-novelty segment lengths: {novelty_lengths}")

        # Convert segment boundaries to timestamps
        # Segments are (start_inclusive, end_exclusive), with consecutive segments
        # sharing their boundary point. We only need the start of each segment.
        boundary_bars = set()
        for start, end in segments:
            boundary_bars.add(start)

        # Sort boundary bars
        boundary_bars = sorted(boundary_bars)

        # Convert to cuepoints
        cuepoints: CuepointList = [
            Cuepoint(timestamp=first_beat_timestamps[bar_idx], label="")
            for bar_idx in boundary_bars
            if bar_idx < len(first_beat_timestamps)
        ]

        if self.params.debug_mode:
            print(f"CBM cuepoints: {[cp.timestamp for cp in cuepoints]}")

        # Apply heuristics
        # Skip bar 0 boundary (song start) since CBM naturally includes it
        if cuepoints and cuepoints[0].timestamp == first_beat_timestamps[0]:
            cuepoints = cuepoints[1:]

        for heuristic in [
            FirstBeatsOnly,
            SongStartCuepoint,
            SongEndCuepoint,
            DPMeasureAlignment,
        ]:
            cuepoints = heuristic.apply(first_beat_timestamps, cuepoints)

        return cuepoints
