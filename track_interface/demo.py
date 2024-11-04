from track_interface import TrackInterface
from pyrekordbox import Rekordbox6Database

def get_first_beat(beat_grid):
    for beat, _, msec in beat_grid:
        if beat == 1:
            return msec
    return 0

def main():
    db = Rekordbox6Database()
    cuepoint_playlist = db.get_playlist(Name="autocuepoints").one()

    for song in cuepoint_playlist.Songs:
        print(song.Content.Title)
        ti = TrackInterface(song, db)
        ti.clear_hot_cues()
        first_beat = get_first_beat(ti.read_beat_grid())
        cue = ti.get_djmd_cue(timestamp=first_beat)
        ti.add_hot_cues([cue])

if __name__ == "__main__":
    main()