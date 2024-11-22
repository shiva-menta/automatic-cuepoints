from .cuepoint_engine import CuepointEngine
from typing import List


class ChangePointEngine(CuepointEngine):
    def _get_min_changepoint_distance(self, sample_size_msecs: float) -> int:
        """
        We only want change points to be at the first beat of a measure, so we want
        to calculate the number of samples that fit into one measure to set the min
        distance between two change points.
        """
        timestamps = self._get_first_beat_timestamps()
        if len(timestamps) < 2:
            raise ValueError("Not enough measures in song.")
        measure_msecs = timestamps[1] - timestamps[0]

        return int(measure_msecs / sample_size_msecs)

    def _get_closest_timestamp(self, timestamps: List[int], tgt: int) -> int:
        """
        Finds the closest number to tgt in nums.
        """
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

    def _convert_changepoints_to_first_beats(
        self, change_points: List[int]
    ) -> List[int]:
        """
        Calculated change points aren't guaranteed to be at the first beat of a measure,
        so this function finds the closest first beats of a measure to all change points and
        returns them.
        """
        timestamps = self._get_first_beat_timestamps()
        return [
            self._get_closest_timestamp(timestamps, change_point)
            for change_point in change_points
        ]

    def _post_process(self, change_points: List[int]) -> List[int]:
        """
        1. Add cue point for first beat of first measure (unless there's already a cuepoint at the first beat of second measure).
        2. Remove last cue point if it is one of last four beats total.

        Some heuristic for track intros
        """
        first_beat = self._get_first_beat_timestamps()[0]
        last_measure_beat_grid = self.beat_grid[-4:]
        if any(map(lambda x: x[2] == change_points[-1], last_measure_beat_grid)):
            change_points.pop()

        return [first_beat] + change_points
