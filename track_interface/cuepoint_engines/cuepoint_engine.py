from typing import List, Tuple


class CuepointEngine:
    def __init__(self, file_path: str, beat_grid: List[Tuple[int, float, int]], params):
        self.file_path = file_path
        self.beat_grid = beat_grid
        self.params = params if params else self._get_default_params()

    def generate_cuepoints(self) -> List[int]:
        """
        Generates cuepoint placement in milliseconds.
        """
        raise NotImplementedError("Base class function not implemented.")

    def _get_first_beat_timestamps(self) -> List[int]:
        return [beat_tuple[2] for beat_tuple in self.beat_grid if beat_tuple[0] == 1]

    def _get_default_params(self):
        """
        Gets parameters to be used in model to facilitate fine tuning.
        """
        raise NotImplementedError("Base class function not implemented.")
