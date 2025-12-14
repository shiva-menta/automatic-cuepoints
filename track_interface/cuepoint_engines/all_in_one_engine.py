from track_interface.cuepoint_engines.heuristics import (
    FirstBeatsOnly,
    RestrictedMeasureIncrements,
    SongEndCuepoint,
    SongStartCuepoint,
)
from track_interface.cuepoint_engines.cuepoint_engine import BeatGrid, CuepointEngine
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
import modal
import tempfile
import requests

# need to install NATTEN
# ffmpeg isnt needed if using .wav files

app = modal.App("automatic-cuepoints")

allin1_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    # Install git for GitHub installations
    .apt_install("git")
    # PyTorch with CUDA support (stable version - using 2.5.0 for NATTEN compatibility)
    .pip_install(
        "torch==2.5.0",
        "torchvision==0.20.0",
        "torchaudio==2.5.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    # Install NATTEN with matching CUDA version (0.17.5 is the available version for torch 2.5.0)
    .pip_install(
        "natten==0.17.5+torch250cu124",
        find_links="https://whl.natten.org"
    )
    # Install madmom from GitHub (as recommended by allin1)
    .run_commands("pip install git+https://github.com/CPJKU/madmom")
    # Install allin1 from PR #36 branch with NATTEN compatibility fixes
    .run_commands("pip install git+https://github.com/docker-audio-tools/all-in-one-docker-gpu-job.git@fix-DockerGPU")
)


@app.function(image=allin1_image)
def process_audio(audio_bytes: bytes) -> dict:
    """Internal function to process audio and return segments."""
    import allin1
    import tempfile
    import os

    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = allin1.analyze(tmp_path)
        segments = result.segments
        return {"segments": list(segments)}
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.function()
@modal.asgi_app()
def fastapi_app():
    """Web endpoint for the allin1 segmentation service."""
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse

    web_app = FastAPI()

    @web_app.post("/")
    async def find_segments(request: Request):
        audio_bytes = await request.body()
        # Call the Modal function
        result = process_audio.remote(audio_bytes)
        return JSONResponse(content=result)

    return web_app


@dataclass(frozen=True)
class AllInOneEngineParams:
    # If set in debug mode, more print logs will be emitted
    debug_mode: bool = False
    # If modal endpoint url is not set, assume local model run
    modal_endpoint_url: Optional[str] = None


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

    def run_local_model(self, audio_bytes) -> List[float]:
        import allin1

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = allin1.analyze(tmp_path)
            segments = result.segments
            return list(segments)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def run_modal_model(self, audio_bytes) -> List[float]:
        if self.params.modal_endpoint_url is None:
            raise ValueError(
                "modal_endpoint_url not set. Please deploy the Modal app and set the endpoint URL in AllInOneEngineParams. "
                "Deploy with: modal deploy track_interface/cuepoint_engines/all_in_one_engine.py"
            )

        response = requests.post(
            self.params.modal_endpoint_url,
            data=audio_bytes,
            headers={"Content-Type": "application/octet-stream"}
        )
        response.raise_for_status()
        return response.json()["segments"]

    def generate_cuepoints(self, file_path: str, beat_grid: BeatGrid) -> List[int]:
        """
        Generates cuepoint placement in milliseconds.

        Args:
            file_path: Path to the audio file
            beat_grid: Beat grid information for the track

        Returns:
            List of cuepoint timestamps in milliseconds
        """
        # Assert filepath is WAV
        if not file_path.lower().endswith('.wav'):
            file_path = "/Users/shivamenta/Desktop/antiup.wav"
            # raise ValueError(f"File must be WAV format, got: {file_path}")

        # Convert file_path to bytes
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()

        # Use Modal endpoint if URL is set, otherwise run locally
        if self.params.modal_endpoint_url:
            segments_seconds = self.run_modal_model(audio_bytes)
        else:
            segments_seconds = self.run_local_model(audio_bytes)

        # Convert segments from seconds to milliseconds
        segments_ms = [segment * 1000.0 for segment in segments_seconds]

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
        snapped_cuepoints = sorted(list(set(snapped_cuepoints)))

        return snapped_cuepoints

    def _get_default_params(self) -> AllInOneEngineParams:
        """
        Gets default parameters to be used in model to facilitate fine tuning.
        """
        return AllInOneEngineParams()
