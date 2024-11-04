from track_interface import TrackInterface
from pyrekordbox import Rekordbox6Database


def create_cuepoint_at_first_timestamp():
    # get beatgrid for current song

    # create a timestamp for the song
    # return it
    pass

def main():
    db = Rekordbox6Database()
    cuepoint_playlist = db.get_playlist(Name="autocuepoints").one()

    for song in cuepoint_playlist.Songs:
        ti = TrackInterface(song, db)
        print(ti.read_beat_grid())
        break
        # db.clear_cuepoints(song)
    
    # need to access ContentFile for song

    # ti = TrackInterface()
    # path = "/Users/shivamenta/Library/Pioneer/rekordbox/share/PIONEER/USBANLZ/ca0/66f32-6871-4bd0-8d2e-fb6625dfcf7a/ANLZ0000.DAT"
    # ti.read_beat_grid(path)


# given a song object
# want to be able to get the beat grid of the song
# want to be able to clear cuepoints
# 

if __name__ == "__main__":
    main()