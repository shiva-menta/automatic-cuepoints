from typing import List, Tuple
from pyrekordbox.anlz import AnlzFile
import numpy as np

class TrackInterface():
    """
    Custom class for interacting with all data related to a specific song.
    """
    def __init__(self, song, db):
        """
        Song needs to be of format DjmdSongPlaylist.
        """
        self.song = song
        self.song_content = song.Content
        self.content_id = self.song_content.ID
        self.db = db
    
    def read_beat_grid(self) -> List[Tuple[int, float, float]]:
        # Tuple[0] = Beat Number (1-4)
        # Tuple[1] = Tempo
        # Tuple[2] = Time (ms)
        query = self.db.get_content_file(ContentID=self.content_id)
        if query.count() == 0:
            raise ValueError("No content files associated with song.")

        local_file_path = ""
        for result in query.all():
            if ".DAT" in result.rb_local_path:
                local_file_path = result.rb_local_path
                break
        
        if not local_file_path:
            raise ValueError("No ANLZ files associated with given song")

        anlz = AnlzFile.parse_file(local_file_path)
        beat_grid = anlz.get("beat_grid")
        
        # might want to revisit these timestamps (numpy conversion loses some specificity)
        return list(zip(beat_grid[0].tolist(), beat_grid[1].tolist(), beat_grid[2].tolist()))

    def read_hot_cues(self):
        pass

    def clear_hot_cues(self):
        pass

    def write_hot_cue(self):
        pass