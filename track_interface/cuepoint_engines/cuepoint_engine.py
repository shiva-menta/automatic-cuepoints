from typing import List, Tuple

BeatGrid = List[Tuple[int, float, int]]


class CuepointEngine:
    def __init__(self, params):
        self.params = params if params else self._get_default_params()

    def generate_cuepoints(self, file_path: str, beat_grid: BeatGrid) -> List[int]:
        """
        Generates cuepoint placement in milliseconds.
        """
        raise NotImplementedError("Base class function not implemented.")

    def _get_first_beat_timestamps(self, beat_grid: BeatGrid) -> List[int]:
        return [beat_tuple[2] for beat_tuple in beat_grid if beat_tuple[0] == 1]

    def _get_default_params(self):
        """
        Gets parameters to be used in model to facilitate fine tuning.
        """
        raise NotImplementedError("Base class function not implemented.")
