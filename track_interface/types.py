from typing import List, Tuple

# Beat grid entry: (beat_number 1-4, tempo, timestamp_ms)
BeatGrid = List[Tuple[int, float, int]]

# List of cuepoint timestamps in milliseconds
CuepointList = List[int]
