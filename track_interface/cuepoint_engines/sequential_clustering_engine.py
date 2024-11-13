from typing import List
from cuepoint_engine import CuepointEngine

class SongSegment():
    def __init__(self, timestamp_msecs: int, length: int):
        pass

    def get_metrics(self):
        pass

    def join_segment(self, segment: "SongSegment"):
        pass

class SequentialClusteringEngine(CuepointEngine):
    """
    Sequential Clustering Approach
    --
    Divide song into one measure increments. On each iteration, join two increments that have the least difference in audio features (e.g. volume, frequency).
    Repeat until some entropy measure is no longer valid.

    Join Methods: minimal increase in variance for joining two sections, entropy, gini impurity
    """

    def entropy():
        pass

    def generate_cuepoints(self) -> List[int]:
        return []