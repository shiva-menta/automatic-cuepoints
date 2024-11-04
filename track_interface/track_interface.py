from typing import List, Tuple
from pyrekordbox.anlz import AnlzFile
from pyrekordbox.db6 import DjmdSongPlaylist
import datetime

class TrackInterface():
    """
    Custom class for interacting with all data related to a specific song.
    """
    def __init__(self, song, db):
        """
        Song needs to be of format DjmdSongPlaylist.
        """
        print(type(song), type(db))
        if not isinstance(song, DjmdSongPlaylist):
            raise TypeError("Song arg is not of type DjmdSongPlaylist")

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

    def clear_hot_cues(self):
        now = datetime.datetime.now()

        # Update DjmdCue table
        query = self.db.get_cue(ContentID=self.content_id)
        if query.count() == 0:
            print(f"No cues found for song with content ID: {self.content_id}")
            return
        for entry in query:
            self.db.delete(entry)

        # Update DjmdContent table
        djmd_content_entry = self.db.get_content(ID=self.content_id)
        if not djmd_content_entry:
            raise ValueError("Invalid djmd content query for given song.")
        
        djmd_content_entry.CueUpdated = str(int(djmd_content_entry.CueUpdated) + 1)
        djmd_content_entry.rb_local_usn = self.db.increment_local_usn()
        djmd_content_entry.updated_at = now
        
        # Update ContentCue table
        query = self.db.get_content_cue(ContentID=self.content_id)
        if query.count() != 1:
            raise ValueError("Invalid content cue query for given song.")
        
        content_cue_entry = query.first()
        content_cue_entry.Cues = "[]"
        content_cue_entry.rb_local_usn = self.db.increment_local_usn()
        content_cue_entry.updated_at = now

        self.db.commit()

    def read_hot_cues(self):
        pass

    

    def write_hot_cue(self):
        pass