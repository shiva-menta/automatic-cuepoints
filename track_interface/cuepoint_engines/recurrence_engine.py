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


@dataclass(frozen=True)
class RecurrenceEngineParams:
    sample_rate: int
    hop_length: int
    n_mfcc: int
    diagonal_tolerance: float
    debug_mode: bool
    manual_k: bool


class RecurrenceEngine(ChangePointEngine):
    """
    Recurrence engine that primarily relies on finding 'visual' patterns in
    recurrence matrices using MFCC.
    """
    params: RecurrenceEngineParams

    def __init__(self, params=Optional[RecurrenceEngineParams]):
        if params:
            self.params = params
        else:
            self.params = RecurrenceEngineParams(
                sample_rate=1000,
                hop_length=50,
                n_mfcc=13,
                diagonal_tolerance=0.15,
                debug_mode=False,
                manual_k=False,
            )

    def get_recurrence_matrix(self, file_path: str) -> Any:
        y, sr = librosa.load(file_path, sr=self.params.sample_rate)
        song_length = librosa.get_duration(y=y, sr=sr)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.params.hop_length)

        arguments = {
            "data": mfcc,
            "metric": "cosine",
            "mode": "affinity",
            "sym": True,
        }
        if self.params.manual_k:
            # we should scale based on song length (make some prediction about repeated sections based on song length - try to limit based on this assumption)
            arguments["k"] = 400

        # Compute recurrence matrix using MFCC features (assumes a knn value)
        recurrence_matrix = librosa.segment.recurrence_matrix(**arguments)

        with open("/Users/shivamenta/Desktop/tmp.txt", "w") as f:
            f.writelines(",".join(map(lambda x: str(x), row)) + "\n" for row in recurrence_matrix)

        return recurrence_matrix

    def measure_offset_to_frame(self, beat_grid: BeatGrid, measure_offset: int):
        seconds = self.measure_offset_to_seconds(beat_grid, measure_offset)
        return int(librosa.time_to_frames(
            [seconds],
            sr=self.params.sample_rate,
            hop_length=self.params.hop_length,
        )[0])

    def frames_to_measure_offset(self, beat_grid: BeatGrid, frame_offset: int):
        seconds = librosa.frames_to_time(
            [frame_offset],
            sr=self.params.sample_rate,
            hop_length=self.params.hop_length,
        )[0] * 1000.0
        first_beats = self._get_first_beat_timestamps(beat_grid)
        return bisect.bisect_left(first_beats, seconds)

    def measure_offset_to_seconds(self, beat_grid: BeatGrid, measure_offset: int):
        return self._get_first_beat_timestamps(beat_grid)[measure_offset - 1] / 1000.0

    def find_local_maxima_peaks(self, data, aggregation='sum', cluster_tolerance=2, selection='center'):
        """
        Args:
            selection: How to pick representative x from each cluster
                'max'    - x with highest value (default behavior)
                'center' - middle x in cluster (good for balanced choice)
                'first'  - leftmost x (gives 20 in [20,21])
                'last'   - rightmost x (gives 21 in [20,21])
        """
        x_to_y = defaultdict(list)
        for x, y in data:
            x_to_y[x].append(y)

        x_sorted = sorted(x_to_y.keys())

        if aggregation == 'sum':
            values = {x: sum(x_to_y[x]) for x in x_sorted}
        elif aggregation == 'count':
            values = {x: len(x_to_y[x]) for x in x_sorted}

        # Build clusters
        clusters = []
        for x in x_sorted:
            merged = False
            for cluster in clusters:
                if any(abs(x - cx) <= cluster_tolerance for cx in cluster):
                    cluster.append(x)
                    merged = True
                    break
            if not merged:
                clusters.append([x])

        # Select representative from each cluster
        peaks = []
        for cluster in clusters:
            if selection == 'max':
                peak_x = max(cluster, key=lambda x: values[x])
            elif selection == 'center':
                peak_x = cluster[len(cluster) // 2]
            elif selection == 'first':
                peak_x = cluster[0]
            elif selection == 'last':
                peak_x = cluster[-1]

            peaks.append({'x': peak_x, 'value': values[peak_x], 'cluster': cluster})

        return [peak["x"] for peak in sorted(peaks, key=lambda p: p['value'], reverse=True)]

    def get_first_beat_frame_idxs(self, beat_grid: BeatGrid) -> List[int]:
        first_beats = self._get_first_beat_timestamps(beat_grid)
        first_beats_seconds = [beat / 1000.0 for beat in first_beats]

        return librosa.time_to_frames(
            first_beats_seconds,
            sr=self.params.sample_rate,
            hop_length=self.params.hop_length,
        )

    def find_off_main_diagonals(self, recurrence_matrix: Any, beat_grid: BeatGrid) -> List[int]:
        # TODO
        # 4. Because recurrence matrix is symmetrical, you can just take the first cuepoints and translate
        def measure_offset_to_frame(num_measures):
            return self.measure_offset_to_frame(
                beat_grid=beat_grid,
                measure_offset=num_measures
            )

        _start_frame_idx_offset = 8
        # not supposed to be an offset, but this is a fine enough metric
        _min_diagonal_measure_length = measure_offset_to_frame(4)
        mat_size = len(recurrence_matrix)
        first_beat_frame_idxs = self.get_first_beat_frame_idxs(beat_grid)

        if self.params.debug_mode:
            print(f"Start frame offset (measures): {_start_frame_idx_offset}")
            print(f"Min frame diagonal length (frames): {_min_diagonal_measure_length}")

        # accounts for the fact that not all diagonals are perfectly constructed (sampling issues, small variations, etc.)
        tolerance = int(self.params.diagonal_tolerance * _min_diagonal_measure_length)

        def get_longest_diagonals(recurrence_matrix, y_axes_based=True, tolerance=10) -> List[Tuple[int, int]]:
            num_first_beats = len(first_beat_frame_idxs)
            longest_diagonal = [(-1, 0)] * num_first_beats

            def get_matrix_value(pos1, pos2) -> float:
                y_idx = pos1 + pos2 if y_axes_based else pos2
                x_idx = pos2 if y_axes_based else pos1 + pos2
                return recurrence_matrix[y_idx][x_idx]

            for idx, pos1 in enumerate(first_beat_frame_idxs):
                if idx < _start_frame_idx_offset:
                    continue
                diag_start, zero_count = 0, 0
                max_diag_start, max_diag_len = -1, 0
                for pos2 in range(mat_size - pos1):
                    value = get_matrix_value(pos1, pos2)
                    if value == 0:
                        zero_count += 1
                    while zero_count > tolerance:
                        if get_matrix_value(pos1, diag_start) == 0:
                            zero_count -= 1
                        diag_start += 1
                    curr_diag_len = pos2 - diag_start + 1
                    if curr_diag_len > max_diag_len:
                        max_diag_len = curr_diag_len
                        max_diag_start = diag_start
                longest_diagonal[idx] = (max_diag_start, max_diag_len)

            return longest_diagonal

        longest_diagonal_at_y_idx = get_longest_diagonals(
            recurrence_matrix=recurrence_matrix,
            y_axes_based=True,
            tolerance=tolerance,
        )
        longest_diagonal_at_x_idx = get_longest_diagonals(
            recurrence_matrix=recurrence_matrix,
            y_axes_based=False,
            tolerance=tolerance,
        )

        # filter only valid diagonal lengths
        def filter_for_min_diagonal_length(longest_diagonals):
            return [(-1, 0) if diag_len < _min_diagonal_measure_length else (diag_start, diag_len)
                    for (diag_start, diag_len) in longest_diagonals
                    ]
        longest_diagonal_at_y_idx = filter_for_min_diagonal_length(longest_diagonal_at_y_idx)
        longest_diagonal_at_x_idx = filter_for_min_diagonal_length(longest_diagonal_at_x_idx)

        # get diagonal idxs for graphing
        diagonal_y_idxs = [first_beat_frame_idxs[idx] for idx, (_, diag_len) in enumerate(longest_diagonal_at_y_idx) if diag_len != 0]
        diagonal_x_idxs = [first_beat_frame_idxs[idx] for idx, (_, diag_len) in enumerate(longest_diagonal_at_x_idx) if diag_len != 0]
        diag_starts_and_sizes = set()

        def add_starts_and_sizes(longest_diagonals):
            for (diag_start, diag_len) in longest_diagonals:
                if diag_start == -1:
                    continue
                diag_starts_and_sizes.add(
                    (
                        self.frames_to_measure_offset(
                            beat_grid=beat_grid,
                            frame_offset=diag_start
                        ),
                        self.frames_to_measure_offset(
                            beat_grid=beat_grid,
                            frame_offset=diag_len
                        )
                    )
                )
        add_starts_and_sizes(longest_diagonal_at_y_idx)
        add_starts_and_sizes(longest_diagonal_at_x_idx)

        if self.params.debug_mode:
            print(f"All Measure (Start, Duration) Pairs: {diag_starts_and_sizes}")

        # voting algorithm on measures and lengths
        # basically what I want to do is a peak finding algorithm where x axis is the measure (first value), y axis is the count (second value)
        start_to_sizes = defaultdict(list)
        for (start, size) in diag_starts_and_sizes:
            start_to_sizes[start].append(size)
        peaks = self.find_local_maxima_peaks(list(diag_starts_and_sizes))
        changepoint_starts_and_sizes = []
        for peak in peaks:
            frequencies = start_to_sizes[peak]
            avg_freq = statistics.mean(frequencies)
            closest_multiple_of_4 = math.ceil(avg_freq / 4) * 4
            peak_4 = round(peak / 4) * 4 + 1
            changepoint_starts_and_sizes.append((peak_4, closest_multiple_of_4))

        if self.params.debug_mode:
            print(f"Corrected Measure (Start, Duration) Pairs: {changepoint_starts_and_sizes}")

        # also probably need to not round measures to int (maybe tenth of a decimal place - we lose a lot of precision otherwise)
        # general logic - if you're exact at the power of two measurements (or 1 off), then stick with current power of two
        # otherwise, go to next power of two if you're at least reaching into that territory
        # get most common

        # convert to changepoint timestamps
        timestamps_seconds = []
        for (diag_start, diag_len) in changepoint_starts_and_sizes:
            timestamps_seconds.extend([self.measure_offset_to_seconds(
                beat_grid=beat_grid,
                measure_offset=diag_start
            ), self.measure_offset_to_seconds(beat_grid=beat_grid,
                                              measure_offset=diag_start + diag_len)])
        timestamps_seconds.sort()

        if self.params.debug_mode:
            self.visualize_diagonals(recurrence_matrix, list(diagonal_y_idxs), list(diagonal_x_idxs), timestamps_seconds)

        return [timestamp * 1000.0 for timestamp in timestamps_seconds]

    def visualize_diagonals(
        self,
        recurrence_matrix: Any,
        y_start_indices: List[int] = None,
        x_start_indices: List[int] = None,
        changepoint_timestamps: List[float] = None,
        output_path: str = "/Users/shivamenta/Desktop/example.png",
        y_diagonal_color: str = 'blue',
        x_diagonal_color: str = 'red',
        changepoint_color: str = 'green',
        diagonal_linewidth: float = 2,
        diagonal_alpha: float = 0.3,
        changepoint_linewidth: float = 2,
        changepoint_alpha: float = 0.7
    ) -> None:
        """
        Visualizes a recurrence matrix and overlays slope-1 diagonal lines starting at specified y-axis and/or x-axis indices,
        plus changepoint markers.

        Args:
            recurrence_matrix: The recurrence matrix to visualize
            y_start_indices: List of y-axis frame indices where diagonals should start (optional)
            x_start_indices: List of x-axis frame indices where diagonals should start (optional)
            changepoint_timestamps: List of changepoint timestamps in seconds to mark on both axes (optional)
            output_path: Path to save the output image (default: "/Users/shivamenta/Desktop/example.png")
            y_diagonal_color: Color of the y-axis diagonal lines (default: 'blue')
            x_diagonal_color: Color of the x-axis diagonal lines (default: 'red')
            changepoint_color: Color of the changepoint lines (default: 'green')
            diagonal_linewidth: Width of diagonal lines (default: 2)
            diagonal_alpha: Transparency of diagonal lines (default: 0.3)
            changepoint_linewidth: Width of changepoint lines (default: 2)
            changepoint_alpha: Transparency of changepoint lines (default: 0.7)
        """
        # Create figure
        plt.figure(figsize=(10, 8))

        # Display recurrence matrix
        librosa.display.specshow(
            recurrence_matrix,
            x_axis='time',
            y_axis='time',
            cmap='hot',
            sr=self.params.sample_rate,
            hop_length=self.params.hop_length
        )
        plt.colorbar(label='Affinity')
        plt.title('MFCC Recurrence Matrix with Diagonals')
        plt.xlabel('Time (s)')
        plt.ylabel('Time (s)')

        mat_size = len(recurrence_matrix)
        max_duration = librosa.frames_to_time(mat_size - 1, sr=self.params.sample_rate, hop_length=self.params.hop_length)

        # Plot diagonal lines for each y_start index (diagonals starting from y-axis)
        if y_start_indices:
            for y_start in y_start_indices:
                if 0 <= y_start < mat_size:
                    # Convert frame indices to time
                    y_start_time = librosa.frames_to_time(y_start, sr=self.params.sample_rate, hop_length=self.params.hop_length)

                    # Calculate diagonal endpoints
                    # Diagonal has slope 1: y_time = x_time + y_start_time
                    x_end_time = min(max_duration, max_duration - y_start_time)

                    if x_end_time > 0:
                        x_coords = [0, x_end_time]
                        y_coords = [y_start_time, y_start_time + x_end_time]
                        plt.plot(x_coords, y_coords, color=y_diagonal_color, linestyle='--',
                                 linewidth=diagonal_linewidth, alpha=diagonal_alpha)

        # Plot diagonal lines for each x_start index (diagonals starting from x-axis)
        if x_start_indices:
            for x_start in x_start_indices:
                if 0 <= x_start < mat_size:
                    # Convert frame indices to time
                    x_start_time = librosa.frames_to_time(x_start, sr=self.params.sample_rate, hop_length=self.params.hop_length)

                    # Calculate diagonal endpoints
                    # Diagonal has slope 1: y_time = x_time + offset, where x starts at x_start_time
                    y_end_time = min(max_duration, max_duration - x_start_time)

                    if y_end_time > 0:
                        x_coords = [x_start_time, x_start_time + y_end_time]
                        y_coords = [0, y_end_time]
                        plt.plot(x_coords, y_coords, color=x_diagonal_color, linestyle='--',
                                 linewidth=diagonal_linewidth, alpha=diagonal_alpha)

        # Plot changepoint timestamps as vertical and horizontal lines
        if changepoint_timestamps is not None and len(changepoint_timestamps) > 0:
            for timestamp in changepoint_timestamps:
                # Convert from milliseconds to seconds if needed (check if values are > 1000)
                time_seconds = timestamp / 1000.0 if timestamp > 1000 else timestamp

                # Plot vertical line (constant x)
                plt.axvline(x=time_seconds, color=changepoint_color, linestyle='--',
                            linewidth=changepoint_linewidth, alpha=changepoint_alpha)

                # Plot horizontal line (constant y)
                plt.axhline(y=time_seconds, color=changepoint_color, linestyle='--',
                            linewidth=changepoint_linewidth, alpha=changepoint_alpha)

        plt.tight_layout()

        # Save to file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
        plt.close()

    def get_nearest_neighbors(self, file_path: str) -> Any:
        pass

    def find_significant_novelty_curve_peaks(self, recurrence_matrix: Any, beat_grid: BeatGrid) -> List[int]:
        """
        Uses novelty curve approach along the main diagonal to find any strong signs of musical section changes.
        """
        # generate checkerboard kernel
        def get_gaussian_checkerboard_kernel(kernel_size: int, sigma: float = None):
            M = kernel_size
            if sigma is None:
                sigma = M / 6.0

            # Create 1D Gaussian window
            x = np.arange(M) - (M - 1) / 2.0
            gaussian_1d = np.exp(-0.5 * (x / sigma) ** 2)

            # Create 2D Gaussian window via outer product
            kernel = np.outer(gaussian_1d, gaussian_1d)

            # Apply checkerboard pattern
            kernel[:M//2, :M//2] *= 1
            kernel[M//2:, M//2:] *= 1
            kernel[:M//2, M//2:] *= -1
            kernel[M//2:, :M//2] *= -1
            return kernel

        # this needs to be fixed based on measures / bpm / min measure size / etc. (probably a fine tunable-parameter)
        kernel_size = self.measure_offset_to_frame(beat_grid, 8)
        kernel = get_gaussian_checkerboard_kernel(kernel_size)

        # get novelty graph
        def novelty_times_from_matrix(R_smooth, kernel):
            """
            Computes novelty curve from recurrence matrix using checkerboard kernel convolution.

            Returns:
                tuple: (boundary_times, boundary_frames, novelty_times, novelty_curve)
            """
            # Pad matrix for boundary handling
            pad_size = kernel_size // 2
            R_padded = np.pad(R_smooth, pad_size, mode='constant', constant_values=0)

            # Compute novelty by convolving kernel along diagonal
            original_size = R_smooth.shape[0]
            novelty = np.array([
                np.sum(R_padded[i:i+kernel_size, i:i+kernel_size] * kernel)
                for i in range(original_size)
            ])

            # Clip negative values
            novelty = np.maximum(novelty, 0)

            # Smooth with moving average
            window_size = 50
            novelty = np.convolve(novelty, np.ones(window_size)/window_size, mode='same')

            # Normalize
            novelty = novelty / (novelty.max() + 1e-10)

            # Remove spurious spikes with median filter
            from scipy.ndimage import median_filter
            novelty = median_filter(novelty, size=15)

            # Convert frame indices to time
            novelty_times = librosa.frames_to_time(
                np.arange(len(novelty)),
                sr=self.params.sample_rate,
                hop_length=self.params.hop_length
            )

            # Detect peaks with restrictive thresholds (want to be super restrictive here)
            prominence_threshold = np.percentile(novelty, 75) * 0.6

            peaks, _ = find_peaks(
                novelty,
                distance=50,
                prominence=prominence_threshold,
                width=5,
                height=0.3
            )

            # Convert peak frames to timestamps
            boundary_times = librosa.frames_to_time(
                peaks,
                sr=self.params.sample_rate,
                hop_length=self.params.hop_length
            )

            return boundary_times, peaks, novelty_times, novelty

        boundary_times_novelty, boundary_frames_novelty, novelty_times, novelty = novelty_times_from_matrix(recurrence_matrix, kernel)

        # correct measure counts
        if self.params.debug_mode:
            self.visualize_diagonals(recurrence_matrix, changepoint_timestamps=boundary_times_novelty,
                                     output_path="/Users/shivamenta/Desktop/novelty.png")

        # find very strong peaks
        return []

    def generate_cuepoints(self, file_path: str, beat_grid: BeatGrid) -> List[int]:
        recurrence_matrix = self.get_recurrence_matrix(file_path)
        if self.params.debug_mode:
            print(f"Matrix size (frames): {len(recurrence_matrix)}")

        # Intra-Section Similarity
        # self.find_significant_novelty_curve_peaks(recurrence_matrix, beat_grid)

        # Inter-Section Similarity
        # change_points = []
        change_points = self.find_off_main_diagonals(recurrence_matrix, beat_grid)

        # Section Merging (account for over-splitting)

        first_beat_timestamps = self._get_first_beat_timestamps(beat_grid)
        for heuristic in [
            FirstBeatsOnly,
            # SongStartCuepoint,
            SongEndCuepoint,
        ]:
            change_points = heuristic.apply(first_beat_timestamps, change_points)

        return change_points
