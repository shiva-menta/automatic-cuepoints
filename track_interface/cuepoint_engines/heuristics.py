import math
from abc import abstractmethod
from typing import Tuple

from track_interface.types import Cuepoint, CuepointList, TimestampList


class Heuristic:
    """
    Abstract base class for applying heuristics to increase change point accuracy.
    """

    @staticmethod
    @abstractmethod
    def apply(
        first_beat_timestamps: TimestampList, cuepoints: CuepointList
    ) -> CuepointList:
        pass


class FirstBeatsOnly(Heuristic):
    """
    Moves cuepoints to the closest first beat of a measure.
    """

    @staticmethod
    def _get_closest_timestamp_to_target(timestamps: TimestampList, tgt: int) -> int:
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
        first_beat_timestamps: TimestampList, cuepoints: CuepointList
    ) -> CuepointList:
        if not cuepoints:
            return []
        return [
            Cuepoint(
                timestamp=FirstBeatsOnly._get_closest_timestamp_to_target(
                    first_beat_timestamps, cuepoint.timestamp
                ),
                label=cuepoint.label,
            )
            for cuepoint in cuepoints
        ]


class RestrictedMeasureIncrements(Heuristic):
    """
    Attempts to adjust change point location based on distance from previous change point.
    - Considers optimal change point distances as 1, 2, or multiple of four. Preference is
      given to powers of two.
    - Adjusts change point if the number of measures moved to reach closest optimal change point
      is within MEASURE_ADJUSTMENT_TOLERANCE.
    """

    MEASURE_ADJUSTMENT_TOLERANCE = 2

    @staticmethod
    def closest_powers_of_two(prev_measure: int, curr_measure: int) -> Tuple[int, int]:
        diff = curr_measure - prev_measure
        if diff <= 0:
            return ()

        base_log = int(math.log(diff, 2))
        return (prev_measure + 2**base_log, prev_measure + 2 ** (base_log + 1))

    @staticmethod
    def apply(
        first_beat_timestamps: TimestampList, cuepoints: CuepointList
    ) -> CuepointList:
        if not cuepoints:
            return []
        timestamp_to_measure = {
            timestamp: idx for idx, timestamp in enumerate(first_beat_timestamps)
        }
        prev_measure = timestamp_to_measure[cuepoints[0].timestamp]
        new_cuepoints = [cuepoints[0]]

        for cuepoint_idx in range(1, len(cuepoints)):
            curr_cuepoint = cuepoints[cuepoint_idx]
            curr_timestamp = curr_cuepoint.timestamp
            curr_measure = timestamp_to_measure[curr_timestamp]

            closest_four_divisor = (curr_measure - prev_measure) // 4
            possible_next_measures = [
                *RestrictedMeasureIncrements.closest_powers_of_two(
                    prev_measure, curr_measure
                ),
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
                possible_next_measure_distances.index(
                    min(possible_next_measure_distances)
                )
            ]

            if (
                abs(best_possible_next_measure - curr_measure)
                <= RestrictedMeasureIncrements.MEASURE_ADJUSTMENT_TOLERANCE
            ):
                adj_curr_measure = best_possible_next_measure

            prev_measure = adj_curr_measure
            adj_curr_timestamp = first_beat_timestamps[adj_curr_measure]

            if adj_curr_timestamp != new_cuepoints[-1].timestamp:
                new_cuepoints.append(Cuepoint(timestamp=adj_curr_timestamp, label=curr_cuepoint.label))

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
        first_beat_timestamps: TimestampList, cuepoints: CuepointList
    ) -> CuepointList:
        if not cuepoints:
            return []
        timestamp_to_measure = {
            timestamp: idx for idx, timestamp in enumerate(first_beat_timestamps)
        }

        start_measure_votes = {
            idx: 0
            for idx in range(4)
            if first_beat_timestamps[idx] < cuepoints[0].timestamp
        }
        if not start_measure_votes:
            return cuepoints

        for cuepoint in cuepoints:
            curr_measure = timestamp_to_measure[cuepoint.timestamp]
            for possible_start in start_measure_votes:
                for divisor in [4, 2]:
                    if (curr_measure - possible_start) % divisor == 0:
                        start_measure_votes[possible_start] += 1
                        break

        best_start = max(start_measure_votes, key=start_measure_votes.get)
        return [Cuepoint(timestamp=first_beat_timestamps[best_start], label="")] + cuepoints


class FirstBeatCuepoint(Heuristic):
    """
    Conditionally adds a cuepoint to the first beat of a song irregardless if the song starts
    at that point.
    """

    @staticmethod
    def apply(
        first_beat_timestamps: TimestampList, cuepoints: CuepointList
    ) -> CuepointList:
        if not cuepoints:
            return []
        if cuepoints[0].timestamp != first_beat_timestamps[0]:
            return [Cuepoint(timestamp=first_beat_timestamps[0], label="")] + cuepoints
        return cuepoints


class SongEndCuepoint(Heuristic):
    """
    Removes ending cuepoint if it marks the last measure of the song.
    """

    @staticmethod
    def apply(
        first_beat_timestamps: TimestampList, cuepoints: CuepointList
    ) -> CuepointList:
        if not cuepoints:
            return []
        if cuepoints[-1].timestamp == first_beat_timestamps[-1]:
            cuepoints.pop()
        return cuepoints


class MergeAdjacentLabels(Heuristic):
    """
    Merges adjacent cuepoints if they have non-empty equivalent labels.
    Keeps the first cuepoint of each sequence of matching labels.

    Skips merging if the resulting segment would exceed MAX_SEGMENT_MEASURES,
    to prevent over-merging that creates unnaturally long segments.
    """

    # Maximum segment length in measures. If merging would create a segment
    # longer than this, the merge is skipped.
    MAX_SEGMENT_MEASURES = 32

    @classmethod
    def apply(
        cls, first_beat_timestamps: TimestampList, cuepoints: CuepointList
    ) -> CuepointList:
        if not cuepoints:
            return []

        # Build timestamp to measure mapping for segment length calculation
        timestamp_to_measure = {ts: idx for idx, ts in enumerate(first_beat_timestamps)}
        total_measures = len(first_beat_timestamps)

        merged = [cuepoints[0]]
        for i, cuepoint in enumerate(cuepoints[1:], start=1):
            prev_label = merged[-1].label
            curr_label = cuepoint.label

            # Check if labels match and both are non-empty (potential merge)
            if prev_label and curr_label and prev_label == curr_label:
                # Calculate resulting segment length if we merge
                merged_start_ts = merged[-1].timestamp
                if merged_start_ts in timestamp_to_measure:
                    merged_start_measure = timestamp_to_measure[merged_start_ts]

                    # Find the end of the merged segment (next cuepoint or end of song)
                    if i + 1 < len(cuepoints) and cuepoints[i + 1].timestamp in timestamp_to_measure:
                        segment_end_measure = timestamp_to_measure[cuepoints[i + 1].timestamp]
                    else:
                        segment_end_measure = total_measures

                    segment_length = segment_end_measure - merged_start_measure

                    # Skip merge if segment would be too long
                    if segment_length > cls.MAX_SEGMENT_MEASURES:
                        merged.append(cuepoint)
                        continue

                # Labels match and segment is within limit - skip this cuepoint (merge)
                continue

            # Labels don't match or one is empty - keep the cuepoint
            merged.append(cuepoint)

        return merged


class DPMeasureAlignment(Heuristic):
    """
    Uses dynamic programming to find optimal cuepoint-to-measure assignment
    minimizing total displacement + interval penalty cost.

    Unlike RestrictedMeasureIncrements which greedily assigns each cuepoint,
    this finds the globally optimal assignment across all cuepoints.
    """

    # === Cost Parameters (tuned via US-005 parameter sweep) ===
    MAX_DISPLACEMENT = 2          # max measures a cuepoint can move from original
    DISPLACEMENT_WEIGHT = 1.0     # multiplier for displacement cost (increased from 0.5)

    # Interval costs (lower = preferred)
    COST_POWER_OF_TWO = 0.0       # best: intervals that are powers of 2 (2, 4, 8, 16, ...)
    COST_ONE_OR_TWO = 0.5         # okay: intervals of 1 or 2 measures
    COST_MULTIPLE_OF_FOUR = 1.0   # good: intervals divisible by 4 (but not power of 2)
    COST_IRREGULAR = 3.0          # penalty for other intervals (increased from 2.0)

    @classmethod
    def interval_cost(cls, prev_measure: int, curr_measure: int) -> float:
        """Compute cost for the interval between two consecutive cuepoints."""
        diff = curr_measure - prev_measure
        if diff <= 0:
            return float('inf')  # invalid: must move forward

        # Best: powers of 2
        if diff & (diff - 1) == 0:
            return cls.COST_POWER_OF_TWO
        # Okay: 1 or 2
        if diff in (1, 2):
            return cls.COST_ONE_OR_TWO
        # Good: multiples of 4
        if diff % 4 == 0:
            return cls.COST_MULTIPLE_OF_FOUR
        # Penalty for irregular intervals
        return cls.COST_IRREGULAR

    @classmethod
    def displacement_cost(cls, original: int, assigned: int) -> float:
        """Cost of moving a cuepoint from its original to assigned measure."""
        return abs(original - assigned) * cls.DISPLACEMENT_WEIGHT

    @classmethod
    def apply(
        cls, first_beat_timestamps: TimestampList, cuepoints: CuepointList
    ) -> CuepointList:
        if len(cuepoints) <= 1:
            return cuepoints

        timestamp_to_measure = {ts: idx for idx, ts in enumerate(first_beat_timestamps)}
        original_measures = [timestamp_to_measure[cp.timestamp] for cp in cuepoints]
        num_measures = len(first_beat_timestamps)
        n = len(cuepoints)

        # dp[i][m] = (min_cost, backpointer to previous measure)
        # Only consider measures within MAX_DISPLACEMENT of original for efficiency
        dp = [{} for _ in range(n)]

        # Base case: first cuepoint
        orig_0 = original_measures[0]
        for m in range(
            max(0, orig_0 - cls.MAX_DISPLACEMENT),
            min(num_measures, orig_0 + cls.MAX_DISPLACEMENT + 1),
        ):
            dp[0][m] = (cls.displacement_cost(orig_0, m), None)

        # Fill DP table
        for i in range(1, n):
            orig_i = original_measures[i]
            for m in range(
                max(0, orig_i - cls.MAX_DISPLACEMENT),
                min(num_measures, orig_i + cls.MAX_DISPLACEMENT + 1),
            ):
                disp_cost = cls.displacement_cost(orig_i, m)
                best_cost, best_prev = float('inf'), None

                for prev_m, (prev_cost, _) in dp[i - 1].items():
                    if prev_m >= m:
                        continue  # must be strictly increasing
                    total = prev_cost + disp_cost + cls.interval_cost(prev_m, m)
                    if total < best_cost:
                        best_cost, best_prev = total, prev_m

                if best_prev is not None:
                    dp[i][m] = (best_cost, best_prev)

        # Handle case where no valid path exists
        if not dp[n - 1]:
            return cuepoints

        # Backtrack to find optimal assignment
        best_final_m = min(dp[n - 1].keys(), key=lambda m: dp[n - 1][m][0])

        assigned_measures = [None] * n
        m = best_final_m
        for i in range(n - 1, -1, -1):
            assigned_measures[i] = m
            if i > 0:
                m = dp[i][m][1]

        # Build result
        return [
            Cuepoint(timestamp=first_beat_timestamps[m], label=cuepoints[i].label)
            for i, m in enumerate(assigned_measures)
        ]
