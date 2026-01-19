import datetime
import json
from typing import Dict
from uuid import uuid4

from track_interface.types import BeatGrid, Cuepoint, CuepointList, TimestampList

from pyrekordbox.anlz import AnlzFile
from pyrekordbox.db6 import ContentCue, DjmdCue, DjmdSongPlaylist


class TrackInterface:
    """
    Custom class for interacting with all data related to a specific song.
    """

    def __init__(self, song, db):
        """
        Song needs to be of format DjmdSongPlaylist.
        """
        if not isinstance(song, DjmdSongPlaylist):
            raise TypeError("Song arg is not of type DjmdSongPlaylist.")

        self.song = song
        self.song_content = song.Content
        self.content_id = self.song_content.ID
        self.content_uuid = self.song_content.UUID
        self.db = db

    def read_beat_grid(self) -> BeatGrid:
        # Tuple[0] = Beat Number (1-4)
        # Tuple[1] = Tempo
        # Tuple[2] = Time (sec)
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
        timestamp_float_arr = beat_grid[2] * 1000
        timestamp_int_arr = timestamp_float_arr.astype(int)
        return list(
            zip(
                beat_grid[0].tolist(), beat_grid[1].tolist(), timestamp_int_arr.tolist()
            )
        )

    def clear_hot_cues(self):
        now = datetime.datetime.now()

        # Update DjmdCue table
        query = self.db.get_cue(ContentID=self.content_id)
        if query.count() == 0:
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
        if query.count() > 1:
            raise ValueError(
                f"Invalid content cue query for given song – {query.count()} results."
            )
        if query.count() == 1:
            content_cue_entry = query.first()
            content_cue_entry.Cues = "[]"
            content_cue_entry.rb_local_usn = self.db.increment_local_usn()
            content_cue_entry.updated_at = now

        self.db.commit()

    def _msec_to_frame(self, timestamp: int) -> int:
        return int(timestamp * 150.0 / 1000)

    def _measure_count_to_color(self, measures: int) -> int:
        match measures:
            case 4:
                return 1
            case 8:
                return 22
            case 12:
                return 38
            case 16:
                return 49
            case 24:
                return 60
            case 32:
                return 42
            case _:
                return 32

    def get_djmd_cue(self, timestamp: int, kind: int = 1, color: int = -1):
        # Kind = 4 for some reason is not a hot cue
        id_ = str(self.db.generate_unused_id(DjmdCue))
        uuid = str(uuid4())
        return DjmdCue.create(
            ID=id_,
            ContentID=self.content_id,
            ContentUUID=self.content_uuid,
            UUID=uuid,
            Kind=kind + (1 if kind >= 4 else 0),
            Color=color,
            InMsec=timestamp,
            InFrame=self._msec_to_frame(timestamp),
            InMpegFrame=0,
            InMpegAbs=0,
            OutMsec=-1,
            OutFrame=0,
            OutMpegFrame=0,
            OutMpegAbs=0,
        )

    def get_empty_content_cue(self):
        id_ = str(self.db.generate_unused_id(ContentCue))
        uuid = str(uuid4())
        return ContentCue.create(
            ID=id_,
            ContentID=self.content_id,
            UUID=uuid,
            Cues="[]",
            rb_local_usn=self.db.increment_local_usn(),
        )

    def add_hot_cues(self, hot_cues):
        if not isinstance(hot_cues, list):
            raise TypeError("Did not pass in a list of hot cues.")
        if not all(map(lambda cuepoint: isinstance(cuepoint, DjmdCue), hot_cues)):
            raise TypeError("Cuepoints arg is not valid array of DjmdCue")
        now = datetime.datetime.now()

        # Insert into DjmdCue table
        for cuepoint in hot_cues:
            self.db.add(cuepoint)

        # Update DjmdContent table
        djmd_content_entry = self.db.get_content(ID=self.content_id)
        if not djmd_content_entry:
            raise ValueError("Invalid djmd content query for given song.")

        prev_cue_updated = (
            int(djmd_content_entry.CueUpdated) if djmd_content_entry.CueUpdated else 0
        )
        djmd_content_entry.CueUpdated = str(prev_cue_updated + 1)
        djmd_content_entry.rb_local_usn = self.db.increment_local_usn()
        djmd_content_entry.updated_at = now

        # Update ContentCue table
        incl_keys = [
            "ID",
            "ContentID",
            "InMsec",
            "InFrame",
            "InMpegFrame",
            "InMpegAbs",
            "OutMsec",
            "OutFrame",
            "OutMpegFrame",
            "OutMpegAbs",
            "Kind",
            "Color",
            "ContentUUID",
            "UUID",
            "created_at",
            "updated_at",
        ]
        cuepoint_jsons = []
        for cuepoint in hot_cues:
            cuepoint_jsons.append(
                {
                    key: getattr(cuepoint, key)
                    for key in incl_keys
                    if hasattr(cuepoint, key)
                }
            )
            for timestamp_field in ["created_at", "updated_at"]:
                if timestamp_field not in cuepoint_jsons[-1]:
                    continue
                cuepoint_jsons[-1][timestamp_field] = cuepoint_jsons[-1][
                    timestamp_field
                ].strftime("%Y-%m-%d %H:%M:%S %Z")

        query = self.db.get_content_cue(ContentID=self.content_id)
        if query.count() > 1:
            raise ValueError(
                f"Invalid content cue query for given song – {query.count()} results."
            )
        if query.count() == 0:
            self.db.add(self.get_empty_content_cue())
            self.db.flush()
            query = self.db.get_content_cue(ContentID=self.content_id)

        content_cue_entry = query.first()
        content_cue_entry.Cues = json.dumps(cuepoint_jsons)
        content_cue_entry.rb_local_usn = self.db.increment_local_usn()
        content_cue_entry.updated_at = now

        self.db.commit()

    def read_hot_cues(self) -> CuepointList:
        query = self.db.get_content_cue(ContentID=self.content_id)
        if query.count() > 1:
            raise ValueError(
                f"Invalid content cue query for given song – {query.count()} results."
            )
        if query.count() == 0:
            return []

        content_cue_entry = query.first()
        cues_json = json.loads(content_cue_entry.Cues)
        return [Cuepoint(timestamp=cue_data["InMsec"], label="") for cue_data in cues_json]

    def get_content_filepath(self) -> str:
        djmd_content_entry = self.db.get_content(ID=self.content_id)
        if not djmd_content_entry:
            raise ValueError("Invalid djmd content query for given song.")

        return djmd_content_entry.FolderPath

    def _get_first_beat_timestamps(self) -> TimestampList:
        return [
            beat_tuple[2] for beat_tuple in self.read_beat_grid() if beat_tuple[0] == 1
        ]

    def _color_label_cuepoints(self, cuepoints, labels: list[str]):
        first_beat_timestamps = self._get_first_beat_timestamps()
        timestamp_to_measure = {
            timestamp: idx for idx, timestamp in enumerate(first_beat_timestamps)
        }

        for idx, cuepoint in enumerate(cuepoints):
            section_end_timestamp = (
                cuepoints[idx + 1].InMsec
                if idx + 1 <= len(cuepoints) - 1
                else first_beat_timestamps[-1]
            )
            measure_count = (
                timestamp_to_measure[section_end_timestamp]
                - timestamp_to_measure[cuepoint.InMsec]
            )

            cuepoint.ColorTableIndex = self._measure_count_to_color(measure_count)
            cuepoint.Comment = f"{labels[idx]} ({measure_count})" if labels[idx] else f"{measure_count}-COUNT"

        return cuepoints

    def generate_cuepoints(self, cuepoints: CuepointList) -> None:
        self.clear_hot_cues()

        labels = [cuepoint.label for cuepoint in cuepoints]
        cuepoint_objs = [
            self.get_djmd_cue(cuepoint.timestamp, kind=idx + 1)
            for idx, cuepoint in enumerate(cuepoints)
        ]
        cuepoint_objs = self._color_label_cuepoints(cuepoint_objs, labels)
        self.add_hot_cues(cuepoint_objs)
