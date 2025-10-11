from dataclasses import dataclass
from typing import List, Tuple, Optional
import pickle
import os

import librosa
import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from track_interface.cuepoint_engines.cache import (
    CACHE_ENABLED,
    convert_to_key,
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
class MLEngineParams:
    # Feature extraction parameters
    frequency_bands: List[Tuple[int, int]]
    n_mfcc: int
    n_chroma: int

    # Model parameters
    model_path: str
    prediction_threshold: float

    # XGBoost hyperparameters (for training)
    max_depth: int
    learning_rate: float
    n_estimators: int


class MLEngine(ChangePointEngine):
    def _get_default_params(self) -> MLEngineParams:
        return MLEngineParams(
            frequency_bands=[
                (0, 200),      # Bass
                (201, 600),    # Low mids
                (601, 3000),   # Mids
                (3001, 7000),  # Upper mids
                (7001, 22000)  # Highs
            ],
            n_mfcc=13,
            n_chroma=12,
            model_path="ml_cuepoint_model.pkl",
            prediction_threshold=0.5,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
        )

    def _get_audio_data(self) -> Tuple[np.ndarray, int]:
        """Load audio data with caching."""
        audio_cache_key = convert_to_key("audio_" + self.file_path)
        sr_cache_key = convert_to_key("sr_" + self.file_path)

        y = sr = None
        if CACHE_ENABLED:
            y = get(audio_cache_key)
            sr = get(sr_cache_key)

        if y is None or sr is None:
            y, sr = librosa.load(self.file_path, sr=None)

            if CACHE_ENABLED:
                put(audio_cache_key, y)
                put(sr_cache_key, sr)

        return y, sr

    def _get_stft_data(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Get STFT magnitude and frequency data with caching."""
        mag_cache_key = convert_to_key("mag_" + self.file_path)
        freqs_cache_key = convert_to_key("freq_" + self.file_path)

        magnitude = freqs = None
        if CACHE_ENABLED:
            magnitude = get(mag_cache_key)
            freqs = get(freqs_cache_key)

        y, sr = self._get_audio_data()

        if magnitude is None or freqs is None:
            n_fft = 2048
            stft_result = librosa.stft(y, n_fft=n_fft, hop_length=512)
            magnitude = np.abs(stft_result)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            if CACHE_ENABLED:
                put(mag_cache_key, magnitude)
                put(freqs_cache_key, freqs)

        return magnitude, freqs, sr

    def _extract_measure_features(
        self,
        start_ms: float,
        end_ms: float,
        y: np.ndarray,
        sr: int,
        magnitude: np.ndarray,
        freqs: np.ndarray,
        song_length_ms: float
    ) -> np.ndarray:
        """
        Extract rich features for a single measure.
        Returns a feature vector for the measure.
        """
        # Convert timestamps to sample indices
        start_sample = int((start_ms / 1000.0) * sr)
        end_sample = int((end_ms / 1000.0) * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(y), end_sample)

        # Extract audio segment
        y_segment = y[start_sample:end_sample]

        # Convert to frame indices for STFT
        hop_length = 512
        start_frame = int((start_ms / song_length_ms) * magnitude.shape[1])
        end_frame = int((end_ms / song_length_ms) * magnitude.shape[1])
        start_frame = max(0, start_frame)
        end_frame = min(magnitude.shape[1], end_frame)

        features = []

        # 1. Frequency band energies (5 features)
        for lower, upper in self.params.frequency_bands:
            freq_indices = np.where((freqs >= lower) & (freqs <= upper))[0]
            if start_frame < end_frame and len(freq_indices) > 0:
                band_energy = np.mean(magnitude[freq_indices, start_frame:end_frame])
            else:
                band_energy = 0.0
            features.append(band_energy)

        # 2. Spectral features
        if len(y_segment) > 0:
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=y_segment, sr=sr, hop_length=hop_length)
            features.append(np.mean(centroid))
            features.append(np.std(centroid))

            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr, hop_length=hop_length)
            features.append(np.mean(bandwidth))

            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr, hop_length=hop_length)
            features.append(np.mean(rolloff))

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y_segment, hop_length=hop_length)
            features.append(np.mean(zcr))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # 3. MFCCs (13 coefficients, mean and std = 26 features)
        if len(y_segment) > 0:
            mfccs = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=self.params.n_mfcc, hop_length=hop_length)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
        else:
            features.extend([0.0] * (self.params.n_mfcc * 2))

        # 4. Chroma features (12 features)
        if len(y_segment) > 0:
            chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr, hop_length=hop_length)
            features.extend(np.mean(chroma, axis=1))
        else:
            features.extend([0.0] * self.params.n_chroma)

        # 5. Onset strength
        if len(y_segment) > 0:
            onset_env = librosa.onset.onset_strength(y=y_segment, sr=sr, hop_length=hop_length)
            features.append(np.mean(onset_env))
            features.append(np.max(onset_env))
            features.append(np.std(onset_env))
        else:
            features.extend([0.0, 0.0, 0.0])

        # 6. RMS energy
        if len(y_segment) > 0:
            rms = librosa.feature.rms(y=y_segment, hop_length=hop_length)
            features.append(np.mean(rms))
            features.append(np.std(rms))
        else:
            features.extend([0.0, 0.0])

        # 7. Spectral flux (change from previous measure)
        if start_frame < end_frame:
            if start_frame > 0:
                prev_spectrum = np.mean(magnitude[:, max(0, start_frame-5):start_frame], axis=1)
                curr_spectrum = np.mean(magnitude[:, start_frame:end_frame], axis=1)
                spectral_flux = np.sqrt(np.sum((curr_spectrum - prev_spectrum)**2))
            else:
                spectral_flux = 0.0
        else:
            spectral_flux = 0.0
        features.append(spectral_flux)

        return np.array(features, dtype=np.float32)

    def extract_all_features(self) -> Tuple[np.ndarray, List[int]]:
        """
        Extract features for all measures in the song.
        Returns: (feature_matrix, measure_timestamps)
        """
        first_beat_timestamps = self._get_first_beat_timestamps()

        if len(first_beat_timestamps) < 2:
            return np.array([]), []

        y, sr = self._get_audio_data()
        magnitude, freqs, sr = self._get_stft_data()
        song_length_ms = librosa.get_duration(path=self.file_path) * 1000.0

        all_features = []
        measure_timestamps = []

        # Extract features for each measure
        for i in range(len(first_beat_timestamps) - 1):
            start_ms = first_beat_timestamps[i]
            end_ms = first_beat_timestamps[i + 1]

            features = self._extract_measure_features(
                start_ms, end_ms, y, sr, magnitude, freqs, song_length_ms
            )
            all_features.append(features)
            measure_timestamps.append(start_ms)

        # Handle last measure
        if first_beat_timestamps:
            start_ms = first_beat_timestamps[-1]
            end_ms = song_length_ms
            features = self._extract_measure_features(
                start_ms, end_ms, y, sr, magnitude, freqs, song_length_ms
            )
            all_features.append(features)
            measure_timestamps.append(start_ms)

        return np.array(all_features), measure_timestamps

    def generate_cuepoints(self) -> List[int]:
        """Generate cuepoints using trained ML model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install with: pip install xgboost")

        # Check if model exists
        if not os.path.exists(self.params.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.params.model_path}. "
                "Please train the model first using train_ml_engine.py"
            )

        # Load trained model
        with open(self.params.model_path, 'rb') as f:
            model = pickle.load(f)

        # Extract features
        X, measure_timestamps = self.extract_all_features()

        if len(X) == 0:
            return []

        # Predict cuepoints
        predictions = model.predict_proba(X)[:, 1]  # Probability of being a cuepoint

        # Select measures above threshold
        change_points = [
            measure_timestamps[i]
            for i in range(len(predictions))
            if predictions[i] >= self.params.prediction_threshold
        ]

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
