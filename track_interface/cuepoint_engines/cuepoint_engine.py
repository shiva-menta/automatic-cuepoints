from typing import List, Dict, Tuple


class CuepointEngine:
    def __init__(self, file_path: str, beat_grid: List[Tuple[int, float, int]]):
        self.file_path = file_path
        self.beat_grid = beat_grid

    def generate_cuepoints(self) -> List[int]:
        """
        Generates cuepoint placement in milliseconds.
        """
        raise NotImplementedError("Base class function not implemented.")

    def _get_first_beat_timestamps(self) -> List[int]:
        return [beat_tuple[2] for beat_tuple in self.beat_grid if beat_tuple[0] == 1]
