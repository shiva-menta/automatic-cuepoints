import multiprocessing
import argparse

# Temp Ignore warnings
import warnings
from typing import List, Tuple, Dict

from pyrekordbox import Rekordbox6Database
from tqdm import tqdm

from track_interface.cuepoint_engines.cuepoint_engine import CuepointEngine

from track_interface.cuepoint_engines.stft_change_point_engine import (
    StftChangePointEngine,
    StftChangePointParams,
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

_ENGINE_MAP: Dict[str, CuepointEngine] = {
    "recurrence": RecurrenceEngine,
    "stft": StftChangePointEngine,
    "genai": GenAIEngine,
}

_DEFAULT_TRACK_PATH = "/Users/shivamenta/Desktop/training_data/MPH - One Sixty.mp3"


def _get_cuepoints_worker(song_data: Tuple) -> Tuple[str, List[int]]:
    """Worker function to generate cuepoints for single track."""
    model_str, file_path, beat_grid = song_data
    cuepoint_engine: CuepointEngine = _ENGINE_MAP[model_str](params=None)
    return (file_path, cuepoint_engine.generate_cuepoints(file_path=file_path, beat_grid=beat_grid))


def add_cuepoints_to_test_data(args):
    db = Rekordbox6Database()
    playlist = db.get_playlist(Name="test_data").one()

    songs_data = []
    filepath_to_interface: Dict[str, TrackInterface] = {}
    for song in playlist.Songs:
        ti = TrackInterface(song, db)
        filepath = ti.get_content_filepath()
        filepath_to_interface[filepath] = ti
        songs_data.append((
            args.model,
            filepath,
            ti.read_beat_grid(),
        ))

    if not songs_data:
        raise ValueError("No valid tracks found.")

    num_processes = args.num_processes
    results = []
    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(_get_cuepoints_worker, songs_data),
                total=len(songs_data),
                desc="Processing songs"
            ))
    elif num_processes == 1:
        for song_data in tqdm(songs_data):
            results.append(_get_cuepoints_worker(song_data))

    filepath_to_cuepoints = {filepath: cuepoints for (filepath, cuepoints) in results}
    for _, (filepath, ti) in enumerate(filepath_to_interface.items(), start=1):
        ti.generate_cuepoints(cuepoint_timestamps=filepath_to_cuepoints[filepath])


def get_cuepoint_engine_performance_metrics(
        cuepoint_engine: CuepointEngine, debug_mode: bool, hot_cues, file_path: str, beat_grid) -> Dict[
        str, int]:
    estimated_cuepoints = cuepoint_engine.generate_cuepoints(file_path=file_path,
                                                             beat_grid=beat_grid)
    labeled_cuepoints = hot_cues

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
    if debug_mode:
        print(f"For filepath: {file_path}")
        print(f"* Actual cuepoints: {labeled_cuepoints}")
        print(f"* Predicted cuepoints: {estimated_cuepoints}")
        print(f"* Error matrix: {error_matrix}")
        print("\n")

    return error_matrix


def _process_song_metrics(song_data: Tuple) -> Tuple[str, Dict[str, int]]:
    """Worker function to process a single song's metrics."""
    model_str, debug_mode, file_path, beat_grid, hot_cues = song_data
    # Initialize engine
    params = None
    match model_str:
        case "recurrence":
            params = RecurrenceEngineParams(
                sample_rate=1000,
                hop_length=50,
                n_mfcc=13,
                diagonal_tolerance=0.15,
                debug_mode=debug_mode,
            )

    model_inst = _ENGINE_MAP[model_str](params=params)

    # Get metrics
    metrics = get_cuepoint_engine_performance_metrics(
        model_inst, debug_mode, hot_cues, file_path, beat_grid
    )
    return (file_path, metrics)


def get_error_metrics(args):
    db = Rekordbox6Database()
    cuepoint_playlist = db.get_playlist(Name="training_data").one()

    # Prepare data for parallel processing
    songs_data = []
    for song in cuepoint_playlist.Songs:
        ti = TrackInterface(song, db)
        if args.default_track and ti.get_content_filepath() != _DEFAULT_TRACK_PATH:
            continue
        songs_data.append((
            args.model,
            args.debug,
            ti.get_content_filepath(),
            ti.read_beat_grid(),
            ti.read_hot_cues(),
        ))
    songs_data = songs_data[:min(len(songs_data), args.num_songs)]

    if not songs_data:
        raise ValueError("No valid tracks found.")

    # Calculate Aggregate Error Metrics using parallel processing
    num_processes = args.num_processes
    metrics = {key: 0 for key in ["true_positive", "false_positive", "false_negative"]}

    results = []
    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(_process_song_metrics, songs_data),
                total=len(songs_data),
                desc="Processing songs"
            ))
    elif num_processes == 1:
        for song_data in tqdm(songs_data):
            results.append(_process_song_metrics(song_data))

    # Aggregate results and track per-song metrics
    song_metrics = []
    for file_path, track_metrics in results:
        song_metrics.append((file_path, track_metrics))
        for k, v in track_metrics.items():
            metrics[k] += v

    # Sort by false positives and false negatives
    top_fp = sorted(song_metrics, key=lambda x: x[1]["false_positive"], reverse=True)[:5]
    top_fn = sorted(song_metrics, key=lambda x: x[1]["false_negative"], reverse=True)[:5]

    print(f"Metrics: {metrics}")
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
    parser = argparse.ArgumentParser(description="Entrypoint into testing script for auto-cuepoints.")
    # general params
    parser.add_argument("--mode", type=str, default="calc-metrics",
                        help=f"Available modes are: ['calc-metrics', 'add-cuepoints'].")
    parser.add_argument("--model", type=str, default="recurrence", help=f"Available modes are: ['recurrence', 'stft', 'genai']")
    parser.add_argument("--num-processes", type=int, default=100, help="Number of songs to calculate metrics for.")
    # calc-metrics params
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug logging within engines and force one process.")
    parser.add_argument("--default-track", action="store_true", default=False,
                        help="Runs cuepoint calculation on only default track + add visualizer for certain engines.")
    parser.add_argument("--num-songs", type=int, default=100, help="Number of songs to calculate metrics for.")

    args = parser.parse_args()

    # Validations
    assert args.model in _ENGINE_MAP

    if args.mode == "calc-metrics":
        get_error_metrics(args)
    elif args.mode == "add-cuepoints":
        add_cuepoints_to_test_data(args)
    else:
        raise ValueError("Invalid arguments passed to CLI.")


if __name__ == "__main__":
    main()
