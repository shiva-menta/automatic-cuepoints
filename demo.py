import collections
import math
import multiprocessing

# Temp Ignore warnings
import warnings
from itertools import product
from typing import List, Tuple, Dict

import librosa
import numpy as np
import ruptures as rpt
from pyrekordbox import Rekordbox6Database
from tqdm import tqdm

from track_interface.cuepoint_engines.cuepoint_engine import CuepointEngine

from track_interface.cuepoint_engines.stft_change_point_engine import (
    StftChangePointEngine,
    StftChangePointParams,
)
from track_interface.cuepoint_engines.trends_engine import (
    TrendsEngine,
    TrendsEngineParams
)
from track_interface.cuepoint_engines.genai_engine import (
    GenAIEngine,
    GenAIEngineParams
)
from track_interface.cuepoint_engines.recurrence_engine import (
    RecurrenceEngine,
    RecurrenceEngineParams
)

from track_interface.track_interface import TrackInterface

warnings.filterwarnings("ignore", category=DeprecationWarning)

_DEFAULT_CUEPOINT_ENGINE = RecurrenceEngine


def _process_song_metrics_two(song_data: Tuple) -> Tuple[str, List[int]]:
    """Worker function to generate cuepoints for single track."""
    file_path, beat_grid = song_data
    cuepoint_engine = StftChangePointEngine()
    return (file_path, cuepoint_engine.generate_cuepoints(file_path=file_path, beat_grid=beat_grid))


def add_cuepoints_to_test_data():
    db = Rekordbox6Database()
    playlist = db.get_playlist(Name="test_data").one()

    songs_data = []
    filepath_to_interface: Dict[str, TrackInterface] = {}
    for song in playlist.Songs:
        ti = TrackInterface(song, db)
        filepath = ti.get_content_filepath()
        songs_data.append((
            filepath,
            ti.read_beat_grid(),
        ))
        filepath_to_interface[filepath] = ti

    results = []
    num_processes = multiprocessing.cpu_count() // 2
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(_process_song_metrics_two, songs_data),
            total=len(songs_data),
            desc="Processing songs"
        ))

    filepath_to_cuepoints = {filepath: cuepoints for (filepath, cuepoints) in results}

    for _, (filepath, ti) in enumerate(filepath_to_interface.items(), start=1):
        ti.generate_cuepoints(cuepoint_timestamps=filepath_to_cuepoints[filepath])


def get_cuepoint_engine_performance_metrics(hot_cues, file_path, beat_grid) -> Dict[str, int]:
    cuepoint_engine = _DEFAULT_CUEPOINT_ENGINE()
    estimated_cuepoints = cuepoint_engine.generate_cuepoints(file_path=file_path,
                                                             beat_grid=beat_grid)
    labeled_cuepoints = hot_cues
    # print(f"filepath: {file_path}")
    # print(f"Actual hot cues: {hot_cues}")
    # print(f"Estimated hot cues: {estimated_cuepoints}\n\n")

    estimated_idx = labeled_idx = 0
    tp = fp = fn = 0
    while estimated_idx < len(estimated_cuepoints) or labeled_idx < len(
        labeled_cuepoints
    ):
        est_cp, lab_cp = (
            estimated_cuepoints[estimated_idx] if estimated_idx < len(estimated_cuepoints) else float('inf'),
            labeled_cuepoints[labeled_idx] if labeled_idx < len(labeled_cuepoints) else float('inf'),
        )
        if est_cp == lab_cp:
            tp += 1
            estimated_idx += 1
            labeled_idx += 1
        elif est_cp > lab_cp:
            fn += 1
            labeled_idx += 1
        else:
            fp += 1
            estimated_idx += 1

    error_matrix = {"true_positive": tp, "false_positive": fp, "false_negative": fn}
    # print(error_matrix)

    return error_matrix


def _process_song_metrics(song_data: Tuple) -> Tuple[str, Dict[str, int]]:
    """Worker function to process a single song's metrics."""
    file_path, beat_grid, hot_cues = song_data
    metrics = get_cuepoint_engine_performance_metrics(
        hot_cues, file_path, beat_grid
    )
    return (file_path, metrics)


def get_error_metrics():
    db = Rekordbox6Database()
    cuepoint_playlist = db.get_playlist(Name="training_data").one()

    target_file_path = "/Users/shivamenta/Desktop/training_data/MPH - One Sixty.mp3"

    # Prepare data for parallel processing
    songs_data = []
    for song in cuepoint_playlist.Songs:
        ti = TrackInterface(song, db)
        # if ti.get_content_filepath() != target_file_path:
        #     continue
        songs_data.append((
            ti.get_content_filepath(),
            ti.read_beat_grid(),
            ti.read_hot_cues(),
        ))

    # Calculate Aggregate Error Metrics using parallel processing
    # num_processes = multiprocessing.cpu_count() // 2
    num_processes = 4
    metrics = {key: 0 for key in ["true_positive", "false_positive", "false_negative"]}

    results = []
    # songs_data = songs_data[:10]
    # for song_data in songs_data:
    #     results.append(_process_song_metrics(song_data))

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(_process_song_metrics, songs_data),
            total=len(songs_data),
            desc="Processing songs"
        ))

    # Aggregate results and track per-song metrics
    song_metrics = []
    for file_path, track_metrics in results:
        song_metrics.append((file_path, track_metrics))
        for k, v in track_metrics.items():
            metrics[k] += v

    # Sort by false positives and false negatives
    top_fp = sorted(song_metrics, key=lambda x: x[1]["false_positive"], reverse=True)[:5]
    top_fn = sorted(song_metrics, key=lambda x: x[1]["false_negative"], reverse=True)[:5]

    print(metrics)
    print(f"Score: {get_metrics_f1_score(metrics)}")

    print("\nTop 5 songs with highest false positives:")
    for file_path, track_metrics in top_fp:
        print(f"  FP={track_metrics['false_positive']}: {file_path}")

    print("\nTop 5 songs with highest false negatives:")
    for file_path, track_metrics in top_fn:
        print(f"  FN={track_metrics['false_negative']}: {file_path}")

    return metrics


def get_metrics_f1_score(metrics):
    precision = (
        metrics["true_positive"]
        * 1.0
        / (metrics["true_positive"] + metrics["false_positive"])
        if (metrics["true_positive"] + metrics["false_positive"])
        else 1
    )
    recall = (
        metrics["true_positive"]
        * 1.0
        / (metrics["true_positive"] + metrics["false_negative"])
        if (metrics["true_positive"] + metrics["false_negative"])
        else 1
    )

    return (
        (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    )


def main():
    get_error_metrics()
    # add_cuepoints_to_test_data()


if __name__ == "__main__":
    main()
