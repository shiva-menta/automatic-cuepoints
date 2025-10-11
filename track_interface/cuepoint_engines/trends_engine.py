from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import librosa
import numpy as np

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


class Trend(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    LEVEL = "level"


@dataclass
class MeasureTrends:
    """Represents the trends for each frequency band in a measure."""
    low_trend: Trend
    mid_trend: Trend
    high_trend: Trend

    def matches(self, other: 'MeasureTrends') -> bool:
        """Check if this measure has the same trends as another measure."""
        return (
            self.low_trend == other.low_trend and
            self.mid_trend == other.mid_trend and
            self.high_trend == other.high_trend
        )


@dataclass
class Section:
    """Represents a section of the song with consistent trends."""
    first_measure_idx: int
    last_measure_idx: int
    trends: MeasureTrends


@dataclass(frozen=True)
class TrendsEngineParams:
    low_freq_range: Tuple[int, int]
    mid_freq_range: Tuple[int, int]
    high_freq_range: Tuple[int, int]
    level_threshold: float  # Percentage threshold for considering a trend "level"


class TrendsEngine(ChangePointEngine):
    def _get_default_params(self) -> TrendsEngineParams:
        return TrendsEngineParams(
            low_freq_range=(0, 200),
            mid_freq_range=(201, 3000),
            high_freq_range=(3001, 22000),
            level_threshold=0.20,  # 10% threshold
        )

    def _get_stft_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get STFT magnitude and frequency data, using cache if available.
        Returns: (magnitude, freqs)
        """
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
            if not exists(mag_cache_key):
                put(mag_cache_key, magnitude)
            if not exists(freqs_cache_key):
                put(freqs_cache_key, freqs)

        return magnitude, freqs

    def _sample_frequency_at_timestamp(
        self, timestamp_ms: float, magnitude: np.ndarray, freqs: np.ndarray, song_length_ms: float
    ) -> Tuple[float, float, float]:
        """
        Sample the three frequency bands at a given timestamp.
        Returns: (low_intensity, mid_intensity, high_intensity)
        """
        # Convert timestamp to frame index
        num_frames = magnitude.shape[1]
        frame_idx = int((timestamp_ms / song_length_ms) * num_frames)
        frame_idx = min(frame_idx, num_frames - 1)  # Clamp to valid range

        # Get intensities for each frequency band
        low_indices = np.where(
            (freqs >= self.params.low_freq_range[0]) & (freqs <= self.params.low_freq_range[1])
        )[0]
        mid_indices = np.where(
            (freqs >= self.params.mid_freq_range[0]) & (freqs <= self.params.mid_freq_range[1])
        )[0]
        high_indices = np.where(
            (freqs >= self.params.high_freq_range[0]) & (freqs <= self.params.high_freq_range[1])
        )[0]

        low_intensity = np.sum(magnitude[low_indices, frame_idx])
        mid_intensity = np.sum(magnitude[mid_indices, frame_idx])
        high_intensity = np.sum(magnitude[high_indices, frame_idx])

        return low_intensity, mid_intensity, high_intensity

    def _determine_trend(self, values: List[float]) -> Trend:
        """
        Determine the trend (increasing, decreasing, level) from a list of values.
        Uses linear regression slope to determine trend direction.
        """
        if len(values) < 2:
            return Trend.LEVEL

        # Calculate average values to determine overall trend
        first_val = values[0]
        last_val = values[-1]

        # Avoid division by zero
        if first_val == 0:
            if last_val == 0:
                return Trend.LEVEL
            return Trend.INCREASING if last_val > 0 else Trend.LEVEL

        # Calculate percentage change
        percent_change = (last_val - first_val) / first_val

        if abs(percent_change) < self.params.level_threshold:
            return Trend.LEVEL
        elif percent_change > 0:
            return Trend.INCREASING
        else:
            return Trend.DECREASING

    def _analyze_measure(
        self,
        measure_start_ms: float,
        measure_end_ms: float,
        magnitude: np.ndarray,
        freqs: np.ndarray,
        song_length_ms: float
    ) -> MeasureTrends:
        """
        Analyze a single measure by sampling 3 points and determining trends.
        """
        # Sample 3 points equally dispersed between the beats
        sample_points = np.linspace(measure_start_ms, measure_end_ms, 3, endpoint=False)

        low_samples = []
        mid_samples = []
        high_samples = []

        for timestamp in sample_points:
            low, mid, high = self._sample_frequency_at_timestamp(
                timestamp, magnitude, freqs, song_length_ms
            )
            low_samples.append(low)
            mid_samples.append(mid)
            high_samples.append(high)

        # Determine trends for each frequency band
        low_trend = self._determine_trend(low_samples)
        mid_trend = self._determine_trend(mid_samples)
        high_trend = self._determine_trend(high_samples)

        return MeasureTrends(low_trend, mid_trend, high_trend)

    def _group_measures_into_sections(
        self, measure_trends: List[MeasureTrends]
    ) -> List[Section]:
        """
        Group consecutive measures with matching trends into sections.
        """
        if not measure_trends:
            return []

        sections = []
        current_section = Section(
            first_measure_idx=0,
            last_measure_idx=0,
            trends=measure_trends[0]
        )

        for i in range(1, len(measure_trends)):
            if current_section.trends.matches(measure_trends[i]):
                # Extend current section
                current_section.last_measure_idx = i
            else:
                # Save current section and start new one
                sections.append(current_section)
                current_section = Section(
                    first_measure_idx=i,
                    last_measure_idx=i,
                    trends=measure_trends[i]
                )

        # Don't forget the last section
        sections.append(current_section)

        return sections

    def generate_cuepoints(self) -> List[int]:
        """
        Generate cuepoints based on trend changes in frequency bands.
        """
        # Get first beat timestamps
        first_beat_timestamps = self._get_first_beat_timestamps()

        if len(first_beat_timestamps) < 2:
            return [first_beat_timestamps[0]] if first_beat_timestamps else []

        # Get STFT data
        magnitude, freqs = self._get_stft_data()
        song_length_ms = librosa.get_duration(path=self.file_path) * 1000.0

        # Analyze each measure
        measure_trends = []
        for i in range(len(first_beat_timestamps) - 1):
            measure_start = first_beat_timestamps[i]
            measure_end = first_beat_timestamps[i + 1]
            trends = self._analyze_measure(
                measure_start, measure_end, magnitude, freqs, song_length_ms
            )
            measure_trends.append(trends)

        # Handle the last measure (from last first beat to end of song)
        if first_beat_timestamps:
            last_measure_start = first_beat_timestamps[-1]
            last_measure_end = song_length_ms
            last_trends = self._analyze_measure(
                last_measure_start, last_measure_end, magnitude, freqs, song_length_ms
            )
            measure_trends.append(last_trends)

        # Group measures into sections
        sections = self._group_measures_into_sections(measure_trends)

        # Extract change points (section boundaries)
        change_points = []
        for section in sections:
            # Add the first beat of each section as a change point
            change_points.append(first_beat_timestamps[section.first_measure_idx])

        # Add the end point (first beat after the last section or song end)
        if sections:
            last_section = sections[-1]
            if last_section.last_measure_idx + 1 < len(first_beat_timestamps):
                change_points.append(first_beat_timestamps[last_section.last_measure_idx + 1])
            else:
                # Use song end
                change_points.append(int(song_length_ms))

        # Apply heuristics
        for heuristic in [
            FirstBeatsOnly,
            SongStartCuepoint,
            RestrictedMeasureIncrements,
            SongEndCuepoint,
        ]:
            change_points = heuristic.apply(first_beat_timestamps, change_points)

        return change_points
