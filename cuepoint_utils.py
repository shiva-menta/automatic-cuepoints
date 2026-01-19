"""
Utility functions for cuepoint processing.
Extracted from demo.py for reuse in app.py and CLI.
"""
import multiprocessing
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from pyrekordbox import Rekordbox6Database

from track_interface.cuepoint_engines.cuepoint_engine import CuepointEngine
from track_interface.cuepoint_engines.recurrence_engine import RecurrenceEngine
from track_interface.cuepoint_engines.stft_change_point_engine import StftChangePointEngine
from track_interface.cuepoint_engines.all_in_one_engine import AllInOneEngine
from track_interface.track_interface import TrackInterface
from track_interface.types import CuepointList


# Engine registry
ENGINE_MAP: Dict[str, CuepointEngine] = {
    "recurrence": RecurrenceEngine,
    "stft": StftChangePointEngine,
    "all_in_one": AllInOneEngine,
}


@dataclass
class CuepointProcessingArgs:
    """Arguments for cuepoint processing operations."""
    model: str = "all_in_one"
    num_processes: int = 4
    encryption_key: Optional[str] = None
    num_songs: int = 0  # 0 means no limit (process all songs)


def _get_cuepoints_worker(song_data: Tuple) -> Tuple[str, CuepointList]:
    """Worker function to generate cuepoints for a single track."""
    model_str, file_path, beat_grid = song_data
    cuepoint_engine: CuepointEngine = ENGINE_MAP[model_str](params=None)
    return (file_path, cuepoint_engine.generate_cuepoints(file_path=file_path, beat_grid=beat_grid))


def add_cuepoints_to_playlist(
    playlist_name: str,
    args: CuepointProcessingArgs,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """
    Add cuepoints to all tracks in the specified playlist.

    Args:
        playlist_name: Name of the Rekordbox playlist to process
        args: Processing arguments (model, num_processes, encryption_key)
        progress_callback: Optional callback function to report progress (current, total)

    Returns:
        List of processed file paths

    Raises:
        ValueError: If no valid tracks found or playlist doesn't exist
    """
    if args.encryption_key:
        db = Rekordbox6Database(key=args.encryption_key)
    else:
        db = Rekordbox6Database()

    playlist_query = db.get_playlist(Name=playlist_name)
    if playlist_query.count() == 0:
        raise ValueError(f"Playlist '{playlist_name}' not found")

    playlist = playlist_query.one()

    songs_data = []
    filepath_to_interface: Dict[str, TrackInterface] = {}

    for song in playlist.Songs:
        ti = TrackInterface(song, db)
        filepath = ti.get_content_filepath()
        filepath_to_interface[filepath] = ti
        songs_data.append((
            args.model,
            filepath,
            ti.read_beat_grid(),
        ))
    if args.num_songs > 0:
        songs_data = songs_data[:args.num_songs]
        # Filter filepath_to_interface to match the filtered songs_data
        filtered_filepaths = {song_data[1] for song_data in songs_data}
        filepath_to_interface = {fp: ti for fp, ti in filepath_to_interface.items() if fp in filtered_filepaths}

    if not songs_data:
        raise ValueError("No valid tracks found in playlist.")

    total_songs = len(songs_data)
    num_processes = min(args.num_processes, total_songs)

    results = []
    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            for idx, result in enumerate(pool.imap(_get_cuepoints_worker, songs_data)):
                results.append(result)
                if progress_callback:
                    progress_callback(idx + 1, total_songs)
    else:
        for idx, song_data in enumerate(songs_data):
            results.append(_get_cuepoints_worker(song_data))
            if progress_callback:
                progress_callback(idx + 1, total_songs)

    filepath_to_cuepoints = {filepath: cuepoints for (filepath, cuepoints) in results}
    for filepath, ti in filepath_to_interface.items():
        ti.generate_cuepoints(cuepoints=filepath_to_cuepoints[filepath])

    return list(filepath_to_interface.keys())


def clear_cuepoints_from_playlist(
    playlist_name: str,
    encryption_key: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """
    Clear cuepoints from all tracks in the specified playlist.

    Args:
        playlist_name: Name of the Rekordbox playlist to process
        encryption_key: Optional Rekordbox database encryption key
        progress_callback: Optional callback function to report progress (current, total)

    Returns:
        List of processed file paths

    Raises:
        ValueError: If playlist doesn't exist
    """
    if encryption_key:
        db = Rekordbox6Database(key=encryption_key)
    else:
        db = Rekordbox6Database()

    playlist_query = db.get_playlist(Name=playlist_name)
    if playlist_query.count() == 0:
        raise ValueError(f"Playlist '{playlist_name}' not found")

    playlist = playlist_query.one()

    processed_files = []
    total_songs = len(playlist.Songs)

    for idx, song in enumerate(playlist.Songs):
        ti = TrackInterface(song, db)
        ti.clear_hot_cues()
        processed_files.append(ti.get_content_filepath())

        if progress_callback:
            progress_callback(idx + 1, total_songs)

    return processed_files
