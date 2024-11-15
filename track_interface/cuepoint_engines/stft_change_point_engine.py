from typing import List, Tuple
from .changepoint_engine import ChangePointEngine
import ruptures as rpt
import librosa
import numpy as np

FREQUENCY_BUCKETS = [(0, 200), (201, 600), (601, 3000), (3001, 7000), (7001, 40000)]
PENALTY = 100000000


class StftChangePointEngine(ChangePointEngine):
    def _generate_frequency_data(self) -> List[List[float]]:
        """
        Use STFT (Short Time Fourier Transform) to separate audio signal into XX frequency
        buckets.
        """
        y, sr = librosa.load(self.file_path, sr=None)
        n_fft = 2048
        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=512)
        magnitude = np.abs(stft_result)
        freqs = librosa.fft_frequencies(sr=sr)

        freq_bucket_to_signal = [[] for _ in range(len(FREQUENCY_BUCKETS))]
        num_time_slices = stft_result.shape[1]

        for t in range(num_time_slices):
            for i, (lower, upper) in enumerate(FREQUENCY_BUCKETS):
                freq_indices = np.where((freqs >= lower) & (freqs <= upper))[0]
                bucket_intensity = np.sum(magnitude[freq_indices, t])
                freq_bucket_to_signal[i].append(bucket_intensity)

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
        model = rpt.KernelCPD(kernel="linear", min_size=min_frames).fit(mat)
        change_points = model.predict(pen=PENALTY)

        return [change_point * frame_to_msec for change_point in change_points]

    def _moving_average(self, x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    def _smooth_frequency_data(
        self, freq_bucket_to_signal: List[List[float]]
    ) -> List[List[float]]:
        smoothed_freq_buckets_to_signal = []
        for bucket in freq_bucket_to_signal:
            np_arr = np.array(bucket)
            smoothed_freq_buckets_to_signal.append(self._moving_average(np_arr, 200))

        return smoothed_freq_buckets_to_signal

    def generate_cuepoints(self) -> List[int]:
        freq_bucket_to_signal = self._smooth_frequency_data(
            self._generate_frequency_data()
        )
        song_length = librosa.get_duration(path=self.file_path) * 1000.0
        sample_size_msecs = song_length / len(freq_bucket_to_signal[0])
        change_points = self._change_point_detection(
            freq_bucket_to_signal,
            self._get_min_changepoint_distance(sample_size_msecs),
            sample_size_msecs,
        )
        first_beat_change_points = self._convert_changepoints_to_first_beats(
            change_points
        )

        return first_beat_change_points
