import argparse
from pyrekordbox import Rekordbox6Database
from track_interface.cuepoint_engines.stft_change_point_engine import (
    StftChangePointEngine,
)
from track_interface.track_interface import TrackInterface
from enum import Enum

def get_mode(mode_arg: str) -> str:
    match mode_arg.lower():
        case "clear":
            return "CLEAR"
        case "add":
            return "ADD"
        case _:
            raise ValueError("Invalid mode argument.")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encryption-key")
    parser.add_argument("--playlist")
    parser.add_argument("--mode")
    args = parser.parse_args()

    playlist = args.playlist or "autocuepoints"
    mode = get_mode(args.mode)
    encryption_key = args.encryption_key

    db = (
        Rekordbox6Database(key=encryption_key) if encryption_key else Rekordbox6Database()
    )
    playlist = db.get_playlist(Name=playlist).one()

    for song in playlist.Songs:
        ti = TrackInterface(song, db, StftChangePointEngine)
        if mode == "ADD":
            ti.generate_cuepoints()
        else:
            ti.clear_hot_cues()


if __name__ == "__main__":
    main()
