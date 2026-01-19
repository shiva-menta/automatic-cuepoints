from typing import List, Tuple
from dataclasses import dataclass

# Beat grid entry: (beat_number 1-4, tempo, timestamp_ms)
BeatGrid = List[Tuple[int, float, int]]

# List of timestamps in milliseconds (e.g., first beat timestamps)
TimestampList = List[int]


@dataclass
class Cuepoint:
    timestamp: int
    label: str


# List of cuepoints: (timestamp_ms, label)
CuepointList = List[Cuepoint]
