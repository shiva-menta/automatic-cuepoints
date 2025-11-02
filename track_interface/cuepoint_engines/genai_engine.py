import base64
import io
import os
from dataclasses import dataclass
from typing import List, Literal

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from track_interface.cuepoint_engines.cache import (
    CACHE_ENABLED,
    convert_to_key,
    get,
    put,
)
from track_interface.cuepoint_engines.changepoint_engine import ChangePointEngine
from track_interface.cuepoint_engines.cuepoint_engine import BeatGrid
from track_interface.cuepoint_engines.heuristics import (
    FirstBeatsOnly,
    RestrictedMeasureIncrements,
    SongEndCuepoint,
    SongStartCuepoint,
)
import matplotlib

matplotlib.use('Agg')


@dataclass(frozen=True)
class GenAIEngineParams:
    """Parameters for GenAI-based cuepoint detection."""

    # Recurrence matrix parameters (matching visualizer.ipynb)
    sample_rate: int
    hop_length: int
    n_mfcc: int
    recurrence_metric: str
    recurrence_mode: str

    # Visualization parameters (for image mode)
    gaussian_sigma: float

    # GenAI parameters
    provider: Literal["claude", "gemini"]
    run_mode: Literal["image", "matrix"]


class GenAIEngine(ChangePointEngine):
    """
    Cuepoint engine that uses GenAI (Claude or Gemini) to analyze recurrence matrices
    and detect structural segmentation in audio tracks.

    Supports two providers:
    - claude: Uses Claude Haiku 4.5 (ANTHROPIC_API_KEY)
    - gemini: Uses Gemini Pro (GEMINI_API_KEY)

    Supports two modes:
    - image: Sends a visualization of the recurrence matrix to the model
    - matrix: Sends the raw recurrence matrix data to the model
    """

    def _get_default_params(self) -> GenAIEngineParams:
        return GenAIEngineParams(
            # Recurrence matrix parameters (from visualizer.ipynb)
            sample_rate=1000,
            hop_length=50,
            n_mfcc=13,
            recurrence_metric="cosine",
            recurrence_mode="affinity",

            # Visualization
            gaussian_sigma=1.0,

            # GenAI
            provider="gemini",
            run_mode="matrix",
        )

    def _get_model_config(self):
        """Get model name and API key based on provider."""
        if self.params.provider == "claude":
            return {
                "model": "claude-sonnet-4-5-20250929",
                "api_key_env_var": "CLAUDE_API_KEY"
            }
        elif self.params.provider == "gemini":
            return {
                "model": "gemini-2.5-pro",
                "api_key_env_var": "GEMINI_API_KEY"
            }
        else:
            raise ValueError(f"Invalid provider: {self.params.provider}")

    def _generate_recurrence_matrix(self, file_path: str) -> np.ndarray:
        """
        Generate recurrence matrix from audio file using MFCC features.
        Uses caching to avoid recomputing for the same file.
        """
        cache_key = convert_to_key(f"recurrence_matrix_{file_path}")

        recurrence_matrix = None
        if CACHE_ENABLED:
            recurrence_matrix = get(cache_key)

        if recurrence_matrix is None:
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.params.sample_rate)

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.params.n_mfcc,
                hop_length=self.params.hop_length
            )

            # Compute recurrence matrix
            recurrence_matrix = librosa.segment.recurrence_matrix(
                mfcc,
                metric=self.params.recurrence_metric,
                mode=self.params.recurrence_mode,
                sym=True
            )

            if CACHE_ENABLED:
                put(cache_key, recurrence_matrix)

        return recurrence_matrix

    def _generate_recurrence_matrix_image(
        self,
        recurrence_matrix: np.ndarray,
        file_path: str
    ) -> bytes:
        """
        Generate a PNG image of the recurrence matrix with time labels.
        Returns the image as bytes.
        """
        # Apply Gaussian smoothing
        recurrence_matrix_filtered = gaussian_filter(
            recurrence_matrix,
            sigma=self.params.gaussian_sigma
        )

        # Get song duration for time labels
        song_duration = librosa.get_duration(path=file_path)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Display recurrence matrix
        img = librosa.display.specshow(
            recurrence_matrix_filtered,
            x_axis='time',
            y_axis='time',
            cmap='hot',
            sr=self.params.sample_rate,
            hop_length=self.params.hop_length,
            ax=ax
        )

        plt.colorbar(img, ax=ax, label='Affinity')
        ax.set_title(f'MFCC Recurrence Matrix (Duration: {song_duration:.1f}s)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Time (s)')

        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf.read()

    def _call_genai_with_image(
        self,
        image_bytes: bytes,
        song_duration: float
    ) -> List[float]:
        """
        Call GenAI (Claude or Gemini) with recurrence matrix image and ask for segmentation timestamps.
        Returns list of timestamps in seconds.
        """
        # Get model config
        config = self._get_model_config()

        # Get API key from environment
        api_key = os.getenv(config["api_key_env_var"])
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {config['api_key_env_var']}"
            )

        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Create prompt
        prompt = f"""CRITICAL INSTRUCTION: You MUST return timestamps in SECONDS only. Do NOT use milliseconds or frames.

You are analyzing a recurrence matrix for audio segmentation. This is a self-similarity matrix where bright regions indicate structural similarity in the music.

SONG DURATION: {song_duration:.2f} seconds
The X and Y axes both show time in SECONDS, ranging from 0 to {song_duration:.2f}.

Your task:
1. Identify major structural boundaries in the music (e.g., intro/verse, verse/chorus, chorus/bridge, etc.)
2. Look for strong diagonal lines and checkerboard patterns that indicate repeating sections
3. Return timestamps in SECONDS where significant structural changes occur

IMPORTANT CONSTRAINTS:
- All timestamps must be in SECONDS (decimal numbers)
- All timestamps must be between 0.0 and {song_duration:.2f}
- Use decimal format for precision (e.g., 15.5, not 15500)

Example output format: 0.0, 15.5, 45.2, 78.9, 120.3

Return ONLY the comma-separated timestamps in seconds. No other text or explanation."""
        print(prompt)

        # Call the appropriate API
        if self.params.provider == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            message = client.messages.create(
                model=config["model"],
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            response_text = message.content[0].text.strip()

        elif self.params.provider == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(config["model"])

            # Prepare image for Gemini
            import PIL.Image
            image = PIL.Image.open(io.BytesIO(image_bytes))

            response = model.generate_content([prompt, image])
            response_text = response.text.strip()
        else:
            raise ValueError(f"Invalid provider: {self.params.provider}")

        # Extract timestamps from response
        timestamps = []
        for part in response_text.split(','):
            try:
                timestamp = float(part.strip())
                timestamps.append(timestamp)
            except ValueError:
                continue

        return timestamps

    def _call_genai_with_matrix(
        self,
        recurrence_matrix: np.ndarray,
        file_path: str
    ) -> List[int]:
        """
        Call GenAI (Claude or Gemini) with raw recurrence matrix data and ask for segmentation frames.
        Returns list of frame indices.
        """
        # Get model config
        config = self._get_model_config()

        # Get API key from environment
        api_key = os.getenv(config["api_key_env_var"])
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {config['api_key_env_var']}"
            )

        # Get song info
        song_duration = librosa.get_duration(path=file_path)
        num_frames = recurrence_matrix.shape[0]

        # Sample the matrix to reduce size (GenAI models have token limits)
        # Take every Nth row/column to create a smaller representative matrix
        downsample_factor = max(1, num_frames // 100)
        matrix_sampled = recurrence_matrix[::downsample_factor, ::downsample_factor]

        # Convert to list for JSON serialization
        matrix_list = matrix_sampled.tolist()

        # Create prompt
        prompt = f"""You are analyzing a recurrence matrix for audio segmentation. This is a self-similarity matrix where higher values indicate structural similarity in the music.

Song details:
- Duration: {song_duration:.2f} seconds
- Total frames: {num_frames}
- Matrix size (downsampled): {matrix_sampled.shape[0]}x{matrix_sampled.shape[1]}
- Downsampling factor: {downsample_factor}

The recurrence matrix data is provided below (values between 0-1, higher = more similar):
{matrix_list}

Your task:
1. Identify major structural boundaries in the music by looking at the matrix patterns
2. Look for block structures and transitions in the similarity matrix
3. Return frame indices where significant structural changes occur
4. Remember to scale your frame indices by the downsampling factor of {downsample_factor}

Please provide frame indices as a comma-separated list of integers.
Example: 0, 450, 1200, 2500, 3800

Only return the frame indices, nothing else."""

        # Call the appropriate API
        if self.params.provider == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            message = client.messages.create(
                model=config["model"],
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )
            response_text = message.content[0].text.strip()

        elif self.params.provider == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(config["model"])

            response = model.generate_content(prompt)
            response_text = response.text.strip()
        else:
            raise ValueError(f"Invalid provider: {self.params.provider}")

        # Extract frame indices from response
        frames = []
        for part in response_text.split(','):
            try:
                frame = int(part.strip())
                frames.append(frame)
            except ValueError:
                continue

        return frames

    def generate_cuepoints(self, file_path: str, beat_grid: BeatGrid) -> List[int]:
        """
        Generate cuepoints using GenAI analysis of recurrence matrix.
        Returns list of cuepoint timestamps in milliseconds.
        """
        # Generate recurrence matrix
        recurrence_matrix = self._generate_recurrence_matrix(file_path)

        # Get song duration
        song_duration = librosa.get_duration(path=file_path)

        # Call GenAI based on run mode
        if self.params.run_mode == "image":
            # Generate image and get timestamps
            image_bytes = self._generate_recurrence_matrix_image(
                recurrence_matrix,
                file_path
            )
            timestamps_seconds = self._call_genai_with_image(
                image_bytes,
                song_duration
            )

            # Convert seconds to milliseconds
            change_points = [int(ts * 1000) for ts in timestamps_seconds]
            print(f"Changepoints: {change_points}")

        elif self.params.run_mode == "matrix":
            # Send matrix data and get frame indices
            frame_indices = self._call_genai_with_matrix(
                recurrence_matrix,
                file_path
            )

            # Convert frames to milliseconds using librosa
            timestamps_seconds = librosa.frames_to_time(
                frame_indices,
                sr=self.params.sample_rate,
                hop_length=self.params.hop_length
            )
            change_points = [int(ts * 1000) for ts in timestamps_seconds]

        else:
            raise ValueError(f"Invalid run_mode: {self.params.run_mode}")

        # Apply heuristics
        first_beat_timestamps = self._get_first_beat_timestamps(beat_grid)
        for heuristic in [
            FirstBeatsOnly,
            SongStartCuepoint,
            RestrictedMeasureIncrements,
            SongEndCuepoint,
        ]:
            change_points = heuristic.apply(first_beat_timestamps, change_points)

        return change_points
