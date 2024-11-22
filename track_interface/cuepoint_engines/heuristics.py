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
    prev_measure_num = timestamp_to_measure[cuepoint_timestamps[0]]

    measure_adjustment_tolerance = 1
    for cuepoint_idx in range(1, len(cuepoint_timestamps)):
        curr_timestamp = cuepoint_timestamps[cuepoint_idx]
        curr_measure = timestamp_to_measure[curr_timestamp]
        closest_four_divisor = (curr_measure - prev_measure_num) // 4
        lower, upper = closest_four_divisor * 4, (closest_four_divisor + 1) * 4

        closest_measure = (
            lower if abs(curr_measure - lower) <= abs(curr_measure - upper) else upper
        )
        if abs(curr_measure - closest_measure) <= measure_adjustment_tolerance:
            cuepoint_timestamps[cuepoint_idx] = first_beat_timestamps[closest_measure]

    return cuepoint_timestamps
