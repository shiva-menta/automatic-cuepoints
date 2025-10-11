from dataclasses import dataclass
from typing import List, Tuple

import librosa
import numpy as np
import ruptures as rpt

from track_interface.cuepoint_engines.cache import (
    CACHE_ENABLED,
    convert_to_key,
    exists,
    get,
    put,
)
from track_interface.cuepoint_engines.changepoint_engine import ChangePointEngine
from track_interface.cuepoint_engines.heuristics import (
    FirstBeatsOnly,
    RestrictedMeasureIncrements,
    SongEndCuepoint,
    SongStartCuepoint,
)

from track_interface.cuepoint_engines.cuepoint_engine import BeatGrid


@dataclass(frozen=True)
class StftChangePointParams:
    frequency_buckets: List[Tuple[int, int]]
    frequency_buckets_weights: List[int]
    penalty: int
    model: str
    smoothing_window_size: int


class StftChangePointEngine(ChangePointEngine):
    def _get_default_params(self) -> None:
        return StftChangePointParams(
            frequency_buckets=[
                (0, 200),
                (201, 600),
                (601, 3000),
                (3001, 7000),
                (7001, 22000),
            ],
            frequency_buckets_weights=[1.0, 1.0, 1.0, 0.5, 1.0],
            penalty=75000000,
            model="linear",
            smoothing_window_size=0,
        )

    def _generate_frequency_data(self, file_path) -> List[List[float]]:
        """
        Use STFT (Short Time Fourier Transform) to separate audio signal into XX frequency
        buckets.

        Temporarily adding cache layer.
        """
        mag_cache_key = convert_to_key("mag_" + file_path)
        freqs_cache_key = convert_to_key("freq_" + file_path)

        magnitude = freqs = None
        if CACHE_ENABLED:
            magnitude = get(mag_cache_key)
            freqs = get(freqs_cache_key)

        if magnitude is None or freqs is None:
            y, sr = librosa.load(file_path, sr=None)
            n_fft = 2048
            stft_result = librosa.stft(y, n_fft=n_fft, hop_length=512)
            magnitude = np.abs(stft_result)
            freqs = librosa.fft_frequencies(sr=sr)

        if CACHE_ENABLED:
            if not exists(mag_cache_key):
                put(mag_cache_key, magnitude)
            if not exists(freqs_cache_key):
                put(freqs_cache_key, freqs)

        freq_bucket_to_signal = [[] for _ in range(len(self.params.frequency_buckets))]
        num_time_slices = magnitude.shape[1]

        for t in range(num_time_slices):
            for i, (lower, upper) in enumerate(self.params.frequency_buckets):
                freq_indices = np.where((freqs >= lower) & (freqs <= upper))[0]
                bucket_intensity = np.sum(magnitude[freq_indices, t])
                freq_bucket_to_signal[i].append(
                    bucket_intensity * self.params.frequency_buckets_weights[i]
                )

        return freq_bucket_to_signal

    def _change_point_detection(
        self,
        freq_bucket_to_signal: List[List[float]],
        min_frames: int,
        frame_to_msec: int,
    ) -> List[int]:
        """
        Uses linearly penalized segmentation to find change points in multivariate frequency data.
        """
        mat = np.array(freq_bucket_to_signal).T
        model = rpt.KernelCPD(kernel=self.params.model, min_size=min_frames).fit(mat)
        change_points = model.predict(pen=self.params.penalty)

        return [change_point * frame_to_msec for change_point in change_points]

    def _moving_average(self, x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    def _sma(self, data, window_size):
        smoothed = []
        for i in range(len(data)):
            # Determine the window range
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            window = data[start:end]
            smoothed.append(sum(window) / len(window))
        return smoothed

    def _smooth_frequency_data(
        self, freq_bucket_to_signal: List[List[float]]
    ) -> List[List[float]]:
        """
        Currently unused – setting the right penalty parameter works better than smoothing data.
        """
        smoothed_freq_buckets_to_signal = []
        for bucket in freq_bucket_to_signal:
            smoothed_freq_buckets_to_signal.append(
                self._sma(bucket, self.params.smoothing_window_size)
            )

        return smoothed_freq_buckets_to_signal

    def generate_cuepoints(self, file_path: str, beat_grid: BeatGrid) -> List[int]:
        freq_bucket_to_signal = self._generate_frequency_data(file_path)
        song_length = librosa.get_duration(path=file_path) * 1000.0
        sample_size_msecs = song_length / len(freq_bucket_to_signal[0])
        change_points = self._change_point_detection(
            freq_bucket_to_signal,
            self._get_min_changepoint_distance(sample_size_msecs, beat_grid),
            sample_size_msecs,
        )

        # Heuristics
        first_beat_timestamps = self._get_first_beat_timestamps(beat_grid)
        for heuristic in [
            FirstBeatsOnly,
            SongStartCuepoint,
            RestrictedMeasureIncrements,
            SongEndCuepoint,
        ]:
            change_points = heuristic.apply(first_beat_timestamps, change_points)

        return change_points
