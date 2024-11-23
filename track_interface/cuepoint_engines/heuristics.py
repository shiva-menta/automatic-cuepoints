from typing import List
from abc import abstractmethod
import collections


class Heuristic:
    """
    Abstract base class for applying heuristics to increase change point accuracy.
    """

    @staticmethod
    @abstractmethod
    def apply(
        first_beat_timestamps: List[int], cuepoint_timestamps: List[int]
    ) -> List[int]:
        pass


class FirstBeatsOnly(Heuristic):
    """
    Moves cuepoints to the closest first beat of a measure.
    """

    @staticmethod
    def _get_closest_timestamp_to_target(timestamps: List[int], tgt: int) -> int:
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

    @staticmethod
    def apply(
        first_beat_timestamps: List[int], cuepoint_timestamps: List[int]
    ) -> List[int]:
        return [
            FirstBeatsOnly._get_closest_timestamp_to_target(
                first_beat_timestamps, cuepoint
            )
            for cuepoint in cuepoint_timestamps
        ]


class RestrictedMeasureIncrements(Heuristic):
    """
    Attempts to adjust change point location based on distance from previous change point.
    - Considers optimal change point distances as 1, 2, or multiple of four.
    - Adjusts change point if the number of measures moved to reach closest optimal change point
      is within MEASURE_ADJUSTMENT_TOLERANCE.
    """

    MEASURE_ADJUSTMENT_TOLERANCE = 1

    @staticmethod
    def apply(
        first_beat_timestamps: List[int], cuepoint_timestamps: List[int]
    ) -> List[int]:
        timestamp_to_measure = {
            timestamp: idx for idx, timestamp in enumerate(first_beat_timestamps)
        }
        prev_measure = timestamp_to_measure[cuepoint_timestamps[0]]
        new_cuepoints = [cuepoint_timestamps[0]]

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
                possible_next_measure_distances.index(
                    min(possible_next_measure_distances)
                )
            ]

            if (
                abs(best_possible_next_measure - curr_measure)
                <= RestrictedMeasureIncrements.MEASURE_ADJUSTMENT_TOLERANCE
            ):
                adj_curr_measure = best_possible_next_measure

            adj_curr_timestamp = first_beat_timestamps[adj_curr_measure]

            if adj_curr_timestamp != new_cuepoints[-1]:
                new_cuepoints.append(adj_curr_timestamp)

        return new_cuepoints


class SongStartCuepoint(Heuristic):
    """
    Change point detection doesn't consider the start of a song as a change point.
    Generally, if songs have a one bar intro, unless the penalty parameter is set
    very high, the method fails to detect this one bar intro due to the small window.

    This heuristic adds a change point at the first beat, though it may not always
    align with the first beat itself, as some songs have beatless intros. This is done
    through allowing each existing cuepoint to "vote" for its best guess for song start
    cuepoint, preferring cuepoints that are an even factor of 4 measures away, then 2.
    """

    @staticmethod
    def apply(
        first_beat_timestamps: List[int], cuepoint_timestamps: List[int]
    ) -> List[int]:
        timestamp_to_measure = {
            timestamp: idx for idx, timestamp in enumerate(first_beat_timestamps)
        }

        start_measure_votes = {
            idx: 0
            for idx in range(4)
            if first_beat_timestamps[idx] < cuepoint_timestamps[0]
        }
        if not start_measure_votes:
            return cuepoint_timestamps

        for cuepoint in cuepoint_timestamps:
            curr_measure = timestamp_to_measure[cuepoint]
            for possible_start in start_measure_votes:
                for divisor in [4, 2]:
                    if (curr_measure - possible_start) % divisor == 0:
                        start_measure_votes[possible_start] += 1
                        break

        best_start = max(start_measure_votes, key=start_measure_votes.get)
        return [first_beat_timestamps[best_start]] + cuepoint_timestamps


class FirstBeatCuepoint(Heuristic):
    """
    Conditionally adds a cuepoint to the first beat of a song irregardless if the song starts
    at that point.
    """

    @staticmethod
    def apply(
        first_beat_timestamps: List[int], cuepoint_timestamps: List[int]
    ) -> List[int]:
        if cuepoint_timestamps[0] != first_beat_timestamps[0]:
            return first_beat_timestamps[:1] + cuepoint_timestamps
        return cuepoint_timestamps


class SongEndCuepoint(Heuristic):
    """
    Removes ending cuepoint if it marks the last measure of the song.
    """

    @staticmethod
    def apply(
        first_beat_timestamps: List[int], cuepoint_timestamps: List[int]
    ) -> List[int]:
        if cuepoint_timestamps[-1] == first_beat_timestamps[-1]:
            cuepoint_timestamps.pop()
        return cuepoint_timestamps
