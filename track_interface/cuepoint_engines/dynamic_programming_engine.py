from track_interface.cuepoint_engines.heuristics import (
    FirstBeatsOnly,
    RestrictedMeasureIncrements,
    SongEndCuepoint,
    SongStartCuepoint,
)
from track_interface.cuepoint_engines.cuepoint_engine import BeatGrid
from track_interface.cuepoint_engines.changepoint_engine import ChangePointEngine
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
from scipy.signal import find_peaks


@dataclass(frozen=True)
class DynamicProgrammingEngineParams:
    frames_per_measure: int = 96
    similarity_func: str = "cosine"
    penalty_param: float = 1.0


class DynamicProgrammingEngine(ChangePointEngine):
    """
    Implementation of https://arxiv.org/pdf/2210.15356.
    """

    def __init__(self, params=Optional[DynamicProgrammingEngineParams]):
        if params:
            self.params = params
        else:
            self.params = DynamicProgrammingEngineParams(
                frames_per_measure=96
            )

    def similarity(self, vec1, vec2) -> int:
        pass

    def regularize(self, num_measures: int) -> int:
        if num_measures % 8 == 0:
            return 0
        elif num_measures % 4 == 0:
            return 0.25
        elif num_measures % 2 == 0:
            return 0.5
        return 1

    def score(self, segment_similarity) -> int:
        pass

    def calculate_num_frames(self, beat_grid: List[int]) -> int:
        return self.params.frames_per_measure // 4 * len(beat_grid)

    def get_recurrence_matrix(self, file_path: str) -> Any:
        y, sr = librosa.load(file_path, sr=self.params.sample_rate)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.params.hop_length)

        arguments = {
            "data": mfcc,
            "metric": "cosine",
            "mode": "connectivity",
            "sym": True,
        }
        if self.params.manual_k:
            # we should scale based on song length (make some prediction about repeated sections based on song length - try to limit based on this assumption)
            arguments["k"] = 400

        # Compute recurrence matrix using MFCC features (assumes a knn value)
        recurrence_matrix = librosa.segment.recurrence_matrix(**arguments)

        with open("/Users/shivamenta/Desktop/tmp.txt", "w") as f:
            f.writelines(",".join(map(lambda x: str(x), row)) + "\n" for row in recurrence_matrix)

        return recurrence_matrix

    def generate_cuepoints(self, file_path: str, beat_grid: BeatGrid) -> List[int]:
        return []
