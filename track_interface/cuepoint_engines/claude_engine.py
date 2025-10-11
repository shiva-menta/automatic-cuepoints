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


@dataclass(frozen=True)
class ClaudeEngineParams:
    frequency_buckets: List[Tuple[int, int]]
    frequency_buckets_weights: List[float]
    penalty: int
    model: str


class ClaudeEngine(ChangePointEngine):
    def _get_default_params(self) -> ClaudeEngineParams:
        return ClaudeEngineParams(
            # Similar to STFT engine but with adjusted weights
            frequency_buckets=[
                (0, 200),
                (201, 600),
                (601, 3000),
                (3001, 7000),
                (7001, 22000),
            ],
            frequency_buckets_weights=[1.5, 1.0, 1.0, 0.5, 0.8],
            penalty=90000000,
            model="linear",
        )

    def _get_stft_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get STFT magnitude and frequency data with caching."""
        mag_cache_key = convert_to_key("mag_" + self.file_path)
        freqs_cache_key = convert_to_key("freq_" + self.file_path)

        magnitude = freqs = None
        if CACHE_ENABLED:
            magnitude = get(mag_cache_key)
            freqs = get(freqs_cache_key)

        if magnitude is None or freqs is None:
            y, sr = librosa.load(self.file_path, sr=None)
            n_fft = 2048
            stft_result = librosa.stft(y, n_fft=n_fft, hop_length=512)
            magnitude = np.abs(stft_result)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            if CACHE_ENABLED:
                put(mag_cache_key, magnitude)
                put(freqs_cache_key, freqs)

        return magnitude, freqs

    def _generate_frequency_data(self) -> List[List[float]]:
        """
        Use STFT to separate audio signal into frequency buckets.
        """
        magnitude, freqs = self._get_stft_data()

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
        frame_to_msec: float,
    ) -> List[int]:
        """
        Uses linearly penalized segmentation to find change points.
        """
        mat = np.array(freq_bucket_to_signal).T
        model = rpt.KernelCPD(kernel=self.params.model, min_size=min_frames).fit(mat)
        change_points = model.predict(pen=self.params.penalty)

        return [int(change_point * frame_to_msec) for change_point in change_points]

    def generate_cuepoints(self) -> List[int]:
        """Generate cuepoints using STFT-based change point detection."""
        freq_bucket_to_signal = self._generate_frequency_data()
        song_length = librosa.get_duration(path=self.file_path) * 1000.0
        sample_size_msecs = song_length / len(freq_bucket_to_signal[0])

        change_points = self._change_point_detection(
            freq_bucket_to_signal,
            self._get_min_changepoint_distance(sample_size_msecs),
            sample_size_msecs,
        )

        # Apply heuristics
        first_beat_timestamps = self._get_first_beat_timestamps()
        for heuristic in [
            FirstBeatsOnly,
            SongStartCuepoint,
            RestrictedMeasureIncrements,
            SongEndCuepoint,
        ]:
            change_points = heuristic.apply(first_beat_timestamps, change_points)

        return change_points
