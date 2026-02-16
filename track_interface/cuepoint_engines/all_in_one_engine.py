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
import modal
import hashlib
import soundfile as sf
import io
import librosa


@dataclass(frozen=True)
class AllInOneEngineParams:
    # If set in debug mode, more print logs will be emitted
    debug_mode: bool = False


class AllInOneEngine(CuepointEngine):
    """
    All-in-one engine that combines multiple approaches for cuepoint generation.

    TODO: Add description of the approach and methodology.
    """
    params: AllInOneEngineParams

    def __init__(self, params: Optional[AllInOneEngineParams] = None):
        if params:
            self.params = params
        else:
            self.params = AllInOneEngineParams()
        self.func = modal.Function.from_name("automatic-cuepoints", "process_audio")

    def generate_cuepoints(self, file_path: str, beat_grid: BeatGrid) -> CuepointList:
        """
        Generates cuepoint placement in milliseconds.

        Args:
            file_path: Path to the audio file
            beat_grid: Beat grid information for the track

        Returns:
            List of cuepoint timestamps in milliseconds
        """
        y, sr = librosa.load(file_path, sr=None, mono=False)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, y.T if y.ndim > 1 else y, sr, format='WAV')
        audio_bytes = wav_buffer.getvalue()
        file_name = hashlib.md5(file_path.encode()).hexdigest() + ".wav"

        segments_seconds = self.func.remote(audio_bytes, file_name)
        cuepoints = []

        for segment in segments_seconds:
            start, _, label = segment["start"], segment["end"], segment["label"]
            cuepoints.append(Cuepoint(
                timestamp=start * 1000.0,
                label=label.upper()
            ))

        # Heuristics
        first_beat_timestamps = self._get_first_beat_timestamps(beat_grid)
        for heuristic in [
            FirstBeatsOnly,
            SongStartCuepoint,
            SongEndCuepoint,
            MergeAdjacentLabels,
            DPMeasureAlignment,
        ]:
            cuepoints = heuristic.apply(first_beat_timestamps, cuepoints)

        return cuepoints

    def _get_default_params(self) -> AllInOneEngineParams:
        """
        Gets default parameters to be used in model to facilitate fine tuning.
        """
        return AllInOneEngineParams()
