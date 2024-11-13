from typing import List
from cuepoint_engine import CuepointEngine
import ruptures
import librosa
import numpy as np

FREQUENCY_BUCKETS = [(0, 200), (201, 600), (601, 3000), (3001, 7000), (7001, 40000)]

class ChangePointEngine(CuepointEngine):
    def _generate_frequency_data(self) -> List[List[float]]:
        y, sr = librosa.load(self.file_path, sr=None)
        stft_result = librosa.stft(y, n_fft=2048, hop_length=512)
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

    def _change_point_detection(freq_bucket_to_signal: List[List[float]]) -> List[int]:
        pass

    def generate_cuepoints(self) -> List[int]:
        self._generate_frequency_data()