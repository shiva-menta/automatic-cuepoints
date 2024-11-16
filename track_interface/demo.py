from track_interface import TrackInterface
from pyrekordbox import Rekordbox6Database
from cuepoint_engines.stft_change_point_engine import StftChangePointEngine
from typing import Dict
from tqdm import tqdm


def get_first_beat(beat_grid):
    for beat, _, msec in beat_grid:
        if beat == 1:
            return msec
    return 0

def add_cuepoints_to_test_data():
    db = Rekordbox6Database()
    cuepoint_playlist = db.get_playlist(Name="test_data").one()

    for song in cuepoint_playlist.Songs:
        print(song.Content.Title)
        ti = TrackInterface(song, db, StftChangePointEngine)
        ti.generate_cuepoints()
        return

def get_error_metrics():
    db = Rekordbox6Database()
    cuepoint_playlist = db.get_playlist(Name="training_data").one()

    # Calculate Aggregate Error Metrics
    metrics = {key: 0 for key in ["true_positive", "false_positive", "false_negative"]}
    for song in tqdm(cuepoint_playlist.Songs):
        ti = TrackInterface(song, db, StftChangePointEngine)
        for k, v in ti.get_cuepoint_engine_performance_metrics().items():
            metrics[k] += v
    
    return metrics

# add msec correction function to account for different analysis

def main():
    print(get_error_metrics())

if __name__ == "__main__":
    main()
