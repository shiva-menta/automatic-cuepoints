from typing import List, Dict
from track_interface import TrackInterface

class CuepointEngine():
    def __init__(self, song):
        self.track_inteface = TrackInterface(song)
        self.file_path = self.track_inteface.get_content_filepath()
    
    def generate_cuepoints(self) -> List[int]:
        """
        Generates cuepoint placement in milliseconds.
        """
        raise NotImplementedError("Base class function not implemented.")
    
    def get_labeled_cuepoints(self) -> List[int]:
        return self.track_inteface.read_hot_cues()

    def get_performance_metrics(self) -> Dict[str, int]:
        estimated_cuepoints = self.generate_cuepoints()
        labeled_cuepoints = self.get_labeled_cuepoints()

        print(f"Estimated Cuepoints: {estimated_cuepoints}")
        print(f"Labeled Cuepoints: {labeled_cuepoints}")

        estimated_idx = labeled_idx = 0
        tp = fp = fn = 0
        while estimated_idx < len(estimated_cuepoints) and labeled_idx < labeled_cuepoints:
            est_cp, lab_cp = estimated_cuepoints[estimated_idx], labeled_cuepoints[labeled_idx]
            if est_cp == lab_cp:
                tp += 1
                estimated_idx += 1
                labeled_idx += 1
            elif est_cp > lab_cp:
                fn += 1
                labeled_idx += 1
            else:
                fp += 1
                estimated_idx += 1

        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn
        }