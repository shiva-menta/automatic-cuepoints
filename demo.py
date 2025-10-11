import collections
import math
import multiprocessing

# Temp Ignore warnings
import warnings
from itertools import product
from typing import List, Tuple

import librosa
import numpy as np
import ruptures as rpt
from pyrekordbox import Rekordbox6Database
from tqdm import tqdm

from track_interface.cuepoint_engines.stft_change_point_engine import (
    StftChangePointEngine,
    StftChangePointParams,
)
from track_interface.cuepoint_engines.trends_engine import (
    TrendsEngine,
    TrendsEngineParams
)
from track_interface.cuepoint_engines.ml_engine import (
    MLEngine,
    MLEngineParams
)
from track_interface.track_interface import TrackInterface

warnings.filterwarnings("ignore", category=DeprecationWarning)


def add_cuepoints_to_test_data():
    db = Rekordbox6Database()
    cuepoint_playlist = db.get_playlist(Name="test_data").one()

    for song in tqdm(cuepoint_playlist.Songs):
        ti = TrackInterface(song, db, StftChangePointEngine)
        ti.generate_cuepoints()


def get_error_metrics():
    db = Rekordbox6Database()
    cuepoint_playlist = db.get_playlist(Name="training_data").one()

    # Calculate Aggregate Error Metrics
    metrics = {key: 0 for key in ["true_positive", "false_positive", "false_negative"]}
    for song in tqdm(cuepoint_playlist.Songs):
        ti = TrackInterface(song, db, StftChangePointEngine)
        for k, v in ti.get_cuepoint_engine_performance_metrics().items():
            metrics[k] += v

    print(metrics)
    print(f"Score: {get_metrics_f1_score(metrics)}")
    return metrics


def generate_frequency_bucket_weights(length: int, options=[0.5, 1.0]):
    if length == 1:
        return [[option] for option in options]
    next_results = generate_frequency_bucket_weights(length - 1, options)
    all_weights = set()
    for weights in [
        [option] + result[:] for option in options for result in next_results
    ]:
        mult_factor = int(1.0 / max(weights))
        for idx in range(len(weights)):
            weights[idx] *= mult_factor
        all_weights.add(tuple(weights))

    return [list(weights) for weights in all_weights]


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


def _get_closest_timestamp(timestamps: List[int], tgt: int) -> int:
    closest_idx, closest_dist = 0, abs(tgt - timestamps[0])
    l, r = 0, len(timestamps) - 1

    while l <= r:
        mid = (l + r) // 2
        if abs(timestamps[mid] - tgt) < closest_dist:
            closest_idx, closest_dist = mid, abs(timestamps[mid] - tgt)
        if tgt >= timestamps[mid]:
            l = mid + 1
        else:
            r = mid - 1

    return timestamps[closest_idx]


def _closest_powers_of_two(prev_measure: int, curr_measure: int) -> Tuple[int, int]:
    diff = curr_measure - prev_measure
    if diff <= 0:
        return ()

    base_log = int(math.log(diff, 2))
    return (prev_measure + 2**base_log, prev_measure + 2 ** (base_log + 1))


def get_stft_metrics(
    params, params_num, magnitude, freqs, beat_grid, cuepoints, song_length
):
    first_beat_timestamps = [
        beat_tuple[2] for beat_tuple in beat_grid if beat_tuple[0] == 1
    ]
    freq_bucket_to_signal = [[] for _ in range(len(params.frequency_buckets))]
    num_time_slices = magnitude.shape[1]

    for t in range(num_time_slices):
        for i, (lower, upper) in enumerate(params.frequency_buckets):
            freq_indices = np.where((freqs >= lower) & (freqs <= upper))[0]
            bucket_intensity = np.sum(magnitude[freq_indices, t])
            freq_bucket_to_signal[i].append(
                bucket_intensity * params.frequency_buckets_weights[i]
            )

    # vars for change point detection
    sample_size_msecs = song_length / len(freq_bucket_to_signal[0])
    measure_msecs = first_beat_timestamps[1] - first_beat_timestamps[0]
    min_frames = int(measure_msecs / sample_size_msecs)

    # change point detection
    mat = np.array(freq_bucket_to_signal).T
    model = rpt.KernelCPD(kernel=params.model, min_size=min_frames).fit(mat)
    change_points = model.predict(pen=params.penalty)
    change_points = [change_point * sample_size_msecs for change_point in change_points]

    # first beats only heuristic
    change_points = [
        _get_closest_timestamp(first_beat_timestamps, change_point)
        for change_point in change_points
    ]

    # song start cuepoint heuristic
    timestamp_to_measure = {
        timestamp: idx for idx, timestamp in enumerate(first_beat_timestamps)
    }

    start_measure_votes = {
        idx: 0 for idx in range(4) if first_beat_timestamps[idx] < change_points[0]
    }
    if start_measure_votes:
        for cuepoint in change_points:
            curr_measure = timestamp_to_measure[cuepoint]
            for possible_start in start_measure_votes:
                for divisor in [4, 2]:
                    if (curr_measure - possible_start) % divisor == 0:
                        start_measure_votes[possible_start] += 1
                        break

    best_start = max(start_measure_votes, key=start_measure_votes.get)
    change_points = [first_beat_timestamps[best_start]] + change_points

    # restricted measure increments heuristic
    MEASURE_ADJUSTMENT_TOLERANCE = 1
    timestamp_to_measure = {
        timestamp: idx for idx, timestamp in enumerate(first_beat_timestamps)
    }
    prev_measure = timestamp_to_measure[change_points[0]]
    new_cuepoints = [change_points[0]]

    for cuepoint_idx in range(1, len(change_points)):
        curr_timestamp = change_points[cuepoint_idx]
        curr_measure = timestamp_to_measure[curr_timestamp]

        closest_four_divisor = (curr_measure - prev_measure) // 4
        possible_next_measures = [
            *_closest_powers_of_two(prev_measure, curr_measure),
            prev_measure + 1,
            prev_measure + 2,
            prev_measure + closest_four_divisor * 4,
            prev_measure + (closest_four_divisor + 1) * 4,
        ]
        possible_next_measures = [
            measure
            for measure in possible_next_measures
            if measure < len(first_beat_timestamps)
        ]

        adj_curr_measure = curr_measure
        possible_next_measure_distances = list(
            map(lambda x: abs(x - curr_measure), possible_next_measures)
        )
        best_possible_next_measure = possible_next_measures[
            possible_next_measure_distances.index(min(possible_next_measure_distances))
        ]

        if (
            abs(best_possible_next_measure - curr_measure)
            <= MEASURE_ADJUSTMENT_TOLERANCE
        ):
            adj_curr_measure = best_possible_next_measure

        prev_measure = adj_curr_measure
        adj_curr_timestamp = first_beat_timestamps[adj_curr_measure]

        if adj_curr_timestamp != new_cuepoints[-1]:
            new_cuepoints.append(adj_curr_timestamp)

    change_points = new_cuepoints

    # song end cuepoint heuristic
    if change_points[-1] == first_beat_timestamps[-1]:
        change_points.pop()

    # calculate result metrics
    estimated_idx = labeled_idx = 0
    tp = fp = fn = 0
    while estimated_idx < len(change_points) and labeled_idx < len(cuepoints):
        est_cp, lab_cp = (
            change_points[estimated_idx],
            cuepoints[labeled_idx],
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

    # calculate difference
    return (
        params_num,
        {"true_positive": tp, "false_positive": fp, "false_negative": fn},
    )


def fine_tune_stft():
    """
    Uses multiprocessing to find the best parameters for STFT Change Point Detection.
    Needed to pull out a lot of logic from STFT Change Point Engine in order to compute optimal parameters fast.
    """
    # Multiprocessing DS
    pool = multiprocessing.Pool(processes=6)

    # STFT Params
    frequency_bucket_configs = [
        [(0, 250), (251, 4000), (4001, 22000)],
        [(0, 200), (201, 600), (601, 3000), (3001, 7000), (7001, 22000)],
        [
            (0, 60),
            (61, 250),
            (251, 500),
            (501, 2000),
            (2001, 4000),
            (4001, 6000),
            (6001, 22000),
        ],
    ]
    penalties = list(range(5 * 10**7, 2 * 10**8, 25 * 10**6))
    param_configs = [
        StftChangePointParams(
            frequency_buckets=frequency_buckets,
            frequency_buckets_weights=weight_assn,
            penalty=penalty,
            model="linear",
            smoothing_window_size=0,
        )
        for (frequency_buckets, penalty) in list(
            product(frequency_bucket_configs, penalties)
        )
        for weight_assn in generate_frequency_bucket_weights(len(frequency_buckets))
    ]

    # Open Rekordbox Database
    db = Rekordbox6Database()
    cuepoint_playlist = db.get_playlist(Name="training_data").one()

    print("Launching tasks.")
    param_tasks = []
    for song in tqdm(cuepoint_playlist.Songs):
        ti = TrackInterface(song, db, StftChangePointEngine)
        file_path = ti.get_content_filepath()
        y, sr = librosa.load(file_path, sr=None)
        n_fft = 2048
        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=512)
        magnitude = np.abs(stft_result)
        freqs = librosa.fft_frequencies(sr=sr)

        beat_grid = ti.read_beat_grid()
        hot_cues = ti.read_hot_cues()
        song_duration = librosa.get_duration(path=file_path) * 1000.0

        for param_idx, param_config in enumerate(param_configs):
            param_tasks.append(
                pool.apply_async(
                    get_stft_metrics,
                    (
                        param_config,
                        param_idx,
                        magnitude,
                        freqs,
                        beat_grid,
                        hot_cues,
                        song_duration,
                    ),
                )
            )

    # Await Results
    print("Waiting on processes.")
    results = []
    for async_res in tqdm(param_tasks):
        results.append(async_res.get())

    # Close Multiprocessing Pool
    pool.close()
    pool.join()

    # Extract Results from Queue
    print("Extracting results from queue.")
    agg_res_map = {}
    for param_idx, result_dict in results:
        if param_idx not in agg_res_map:
            agg_res_map[param_idx] = collections.defaultdict(int)
        for k, v in result_dict.items():
            agg_res_map[param_idx][k] += v

    # Find Best Metrics
    best_metrics, best_params, best_score = {}, None, 0
    for param_idx, result_dict in agg_res_map.items():
        score = get_metrics_f1_score(result_dict)
        if score > best_score:
            best_metrics, best_params, best_score = (
                result_dict,
                param_configs[param_idx],
                score,
            )

    # Print Output
    print(best_metrics)
    print(f"Frequency Buckets: {best_params.frequency_buckets}")
    print(f"Frequency Bucket Weights: {best_params.frequency_buckets_weights}")
    print(f"Penalty: {best_params.penalty}")
    print(f"Model: {best_params.model}")
    print(f"Smoothing Window Size: {best_params.smoothing_window_size}")
    print(f"Score: {best_score}")


def main():
    get_error_metrics()
    # add_cuepoints_to_test_data()
    # fine_tune_stft()


if __name__ == "__main__":
    main()
