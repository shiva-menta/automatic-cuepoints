from typing import List


def even_bar_placement(
    first_beat_timestamps: int, cuepoint_timestamps: int
) -> List[int]:
    """
    V1
    Prioritize bars in multiples of four.

    Assumptions:
    - First Bar is Correct
    - some metric of strength that previous placement is correct
    """
    timestamp_to_measure = {
        timestamp: idx for idx, timestamp in enumerate(first_beat_timestamps)
    }
    prev_measure = timestamp_to_measure[cuepoint_timestamps[0]]
    new_cuepoints = [cuepoint_timestamps[0]]

    measure_adjustment_tolerance = 1
    for cuepoint_idx in range(1, len(cuepoint_timestamps)):
        curr_timestamp = cuepoint_timestamps[cuepoint_idx]
        curr_measure = timestamp_to_measure[curr_timestamp]

        closest_four_divisor = (curr_measure - prev_measure) // 4
        possible_next_measures = [
            prev_measure + 1,
            prev_measure + closest_four_divisor * 4,
            prev_measure + 2,
            prev_measure + (closest_four_divisor + 1) * 4,
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
            <= measure_adjustment_tolerance
        ):
            adj_curr_measure = best_possible_next_measure

        adj_curr_timestamp = first_beat_timestamps[adj_curr_measure]

        if adj_curr_timestamp != new_cuepoints[-1]:
            new_cuepoints.append(adj_curr_timestamp)

    return new_cuepoints
