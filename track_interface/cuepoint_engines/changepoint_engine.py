from track_interface.cuepoint_engines.cuepoint_engine import CuepointEngine


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
