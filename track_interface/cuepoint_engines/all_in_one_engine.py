from track_interface.cuepoint_engines.heuristics import (
    FirstBeatsOnly,
    RestrictedMeasureIncrements,
    SongEndCuepoint,
    SongStartCuepoint,
    MergeAdjacentLabels,
    DPMeasureAlignment
)
from track_interface.cuepoint_engines.cuepoint_engine import CuepointEngine
from track_interface.types import BeatGrid, Cuepoint, CuepointList
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import modal
import hashlib
import soundfile as sf
import io
import librosa
import diskcache


@dataclass(frozen=True)
class AllInOneEngineParams:
    # If set in debug mode, more print logs will be emitted
    debug_mode: bool = False
    # When False, bypasses cache (force calculate)
    use_cache: bool = True


class AllInOneEngine(CuepointEngine):
    """
    Cuepoint engine using allin1 segmentation via Modal GPU.
    Post-processes with heuristics for beat alignment and section merging.
    """
    params: AllInOneEngineParams

    def __init__(self, params: Optional[AllInOneEngineParams] = None):
        if params:
            self.params = params
        else:
            self.params = AllInOneEngineParams()
        self.func = modal.Function.from_name("automatic-cuepoints", "process_audio")
        cache_dir = Path.home() / ".cache" / "automatic-cuepoints" / "modal_responses"
        self.cache = diskcache.Cache(str(cache_dir))

    def generate_cuepoints(self, file_path: str, beat_grid: BeatGrid) -> CuepointList:
        """
        Generates cuepoint placement in milliseconds.

        Args:
            file_path: Path to the audio file
            beat_grid: Beat grid information for the track

        Returns:
            List of cuepoint timestamps in milliseconds
        """
        cache_key = hashlib.md5(file_path.encode()).hexdigest()

        # Check cache first if use_cache is True
        if self.params.use_cache and cache_key in self.cache:
            if self.params.debug_mode:
                print(f"Cache hit for {file_path}")
            segments_seconds = self.cache[cache_key]
        else:
            if self.params.debug_mode:
                print(f"Cache miss for {file_path}, calling Modal API")
            y, sr = librosa.load(file_path, sr=None, mono=False)
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, y.T if y.ndim > 1 else y, sr, format='WAV')
            audio_bytes = wav_buffer.getvalue()
            file_name = cache_key + ".wav"

            segments_seconds = self.func.remote(audio_bytes, file_name, not self.params.use_cache)
            # Store result in cache
            self.cache[cache_key] = segments_seconds

        cuepoints = []

        for segment in segments_seconds:
            start, _, label = segment["start"], segment["end"], segment["label"]
            if start == 0:
                # Skip the first cuepoint as we will add this back later in heuristics secion.
                continue
            cuepoints.append(Cuepoint(
                timestamp=start * 1000.0,
                label=label.upper()
            ))

        # Heuristics
        # Ordering validated in US-006: DP before Merge improves F0.75 by +0.0084
        # See scripts/heuristic_ordering_validation.py for analysis
        first_beat_timestamps = self._get_first_beat_timestamps(beat_grid)
        for heuristic in [
            FirstBeatsOnly,
            SongStartCuepoint,
            SongEndCuepoint,
            DPMeasureAlignment,
            MergeAdjacentLabels,
        ]:
            cuepoints = heuristic.apply(first_beat_timestamps, cuepoints)

        return cuepoints

    def _get_default_params(self) -> AllInOneEngineParams:
        """
        Gets default parameters to be used in model to facilitate fine tuning.
        """
        return AllInOneEngineParams()
