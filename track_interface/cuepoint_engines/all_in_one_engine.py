from track_interface.cuepoint_engines.heuristics import (
    FirstBeatsOnly,
    RestrictedMeasureIncrements,
    SongEndCuepoint,
    SongStartCuepoint,
)
from track_interface.cuepoint_engines.cuepoint_engine import CuepointEngine
from track_interface.types import BeatGrid, Cuepoint, CuepointList
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
from collections import defaultdict
from typing import Literal, Any, Optional
from dataclasses import dataclass
import modal
import hashlib

# Modal App Configs
app = modal.App("automatic-cuepoints")
vol = modal.Volume.from_name("cache", create_if_missing=True)
cache_mount_path = "/root/cache"

# Volume filepaths
_OUT_DIR = f"{cache_mount_path}/analyze_outputs/"
_DEMIX_DIR = f"{cache_mount_path}/demix_outputs/"
_SPEC_DIR = f"{cache_mount_path}/spec_outputs/"

allin1_image = (
    modal.Image.from_registry(
        "smenta/automatic-cuepoints:cuda",
    )
)


@app.function(image=allin1_image, gpu="T4", volumes={cache_mount_path: vol})
def process_audio(audio_bytes: bytes, file_name: str, force_recalculate: bool = False) -> list:
    """Internal function to process audio and return segments."""
    import allin1
    import os

    # write audio bytes to file
    tmp_path = f"/tmp/{file_name}"
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    os.environ["TORCH_HOME"] = cache_mount_path

    # get analyze outputs
    result = allin1.analyze(
        tmp_path,
        out_dir=_OUT_DIR,
        demix_dir=_DEMIX_DIR,
        spec_dir=_SPEC_DIR,
        overwrite=force_recalculate,
        keep_byproducts=True,
    )
    segments = result.segments
    simple_segments = [{
        "start": float(segment.start),
        "end": float(segment.end),
        "label": segment.label,
    } for segment in segments]

    # todo(smenta) - figure out if we want to clear out the actual song data in file
    # modal doesn't charge for volume storage yet, but once this is set - we can remove.

    # persist changes
    vol.commit()

    return simple_segments


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
        # Convert file_path to bytes
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        # todo(smenta) - might need to fix based on proper file extension
        file_name = hashlib.md5(file_path.encode()).hexdigest() + ".wav"

        segments_seconds = self.func.remote(audio_bytes, file_name)
        changepoints = []

        for segment in segments_seconds:
            start, end, label = segment["start"], segment["end"], segment["label"]
            for point in [start, end]:
                if not changepoints or point != changepoints[-1]:
                    changepoints.append(point)

        # Convert segments from seconds to milliseconds
        segments_ms = [segment * 1000.0 for segment in changepoints]

        # Snap to beat_grid (find nearest beat for each segment)
        first_beat_timestamps = self._get_first_beat_timestamps(beat_grid)
        snapped_cuepoints = []

        for segment_ms in segments_ms:
            # Find the closest first beat timestamp
            idx = bisect.bisect_left(first_beat_timestamps, segment_ms)

            # Check both the position before and after to find the closest
            closest_beat = None
            if idx == 0:
                closest_beat = first_beat_timestamps[0]
            elif idx == len(first_beat_timestamps):
                closest_beat = first_beat_timestamps[-1]
            else:
                before = first_beat_timestamps[idx - 1]
                after = first_beat_timestamps[idx]
                closest_beat = before if abs(segment_ms - before) < abs(segment_ms - after) else after

            if closest_beat is not None:
                snapped_cuepoints.append(int(closest_beat))

        # Remove duplicates and sort
        change_points = sorted(list(set(snapped_cuepoints)))

        # Convert to CuepointList format with empty labels
        cuepoints: CuepointList = [Cuepoint(timestamp=ts, label="") for ts in change_points]

        # Heuristics
        first_beat_timestamps = self._get_first_beat_timestamps(beat_grid)
        for heuristic in [
            FirstBeatsOnly,
            SongStartCuepoint,
            RestrictedMeasureIncrements,
            SongEndCuepoint,
        ]:
            cuepoints = heuristic.apply(first_beat_timestamps, cuepoints)

        return cuepoints

    def _get_default_params(self) -> AllInOneEngineParams:
        """
        Gets default parameters to be used in model to facilitate fine tuning.
        """
        return AllInOneEngineParams()
