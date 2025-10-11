import sys
import time

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pyrekordbox import Rekordbox6Database, get_config

from track_interface.cuepoint_engines.stft_change_point_engine import (
    StftChangePointEngine,
)
from track_interface.track_interface import TrackInterface


class WorkerThread(QThread):
    progress_changed = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, encryption_key, playlist, mode):
        super().__init__()
        self.encryption_key = encryption_key
        self.playlist = playlist
        self.mode = mode

    def run(self):
        db = (
            Rekordbox6Database(key=self.encryption_key)
            if self.encryption_key
            else Rekordbox6Database()
        )
        playlist = db.get_playlist(Name=self.playlist).one()
        cuepoint_engine = StftChangePointEngine()

        for idx, song in enumerate(playlist.Songs):
            ti = TrackInterface(song, db)
            if self.mode == "Add Cuepoints":
                ti.generate_cuepoints(cuepoint_timestamps=cuepoint_engine.generate_cuepoints(ti.get_content_filepath(),
                                                                                             ti.read_beat_grid()))
            else:
                ti.clear_hot_cues()

            progress = int((idx + 1) * 100.0 / len(playlist.Songs))
            self.progress_changed.emit(progress)

        self.finished.emit()


class AutocuepointsGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Autocuepoints")

        self.layout = QVBoxLayout(self)
        self.form_layout = QVBoxLayout()

        self.encryption_label = QLabel("Encryption Key:", self)
        self.playlist_label = QLabel("Playlist:", self)
        self.action_label = QLabel("Action:", self)
        self.status_label = QLabel("Invalid Input", self)
        self.status_label.setHidden(True)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.resize(300, 40)
        self.progress_bar.setHidden(True)
        # self.debug_label = QLabel(f"{get_config("rekordbox6")}")

        self.encryption_box = QLineEdit(self)
        self.playlist_box = QLineEdit(self)
        self.action_box = QComboBox(self)
        self.action_box.addItem("Clear Cuepoints")
        self.action_box.addItem("Add Cuepoints")

        self.form_layout.addWidget(self.encryption_label)
        self.form_layout.addWidget(self.encryption_box)
        self.form_layout.addWidget(self.playlist_label)
        self.form_layout.addWidget(self.playlist_box)
        self.form_layout.addWidget(self.action_label)
        self.form_layout.addWidget(self.action_box)
        self.form_layout.addWidget(self.status_label)
        self.form_layout.addWidget(self.progress_bar)
        # self.form_layout.addWidget(self.debug_label)

        self.action_button = QPushButton("Take Action", self)
        self.action_button.clicked.connect(self.action)

        self.layout.addLayout(self.form_layout)
        self.layout.addWidget(self.action_button)
        self.show()

    def action(self):
        encryption_key = self.encryption_box.text()
        playlist = self.playlist_box.text()
        mode = self.action_box.currentText()
        self.status_label.setHidden(True)

        try:
            db = (
                Rekordbox6Database(key=encryption_key)
                if encryption_key
                else Rekordbox6Database()
            )
            db.get_playlist(Name=playlist).one()
            self.progress_bar.setValue(0)
            self.progress_bar.setHidden(False)

            self.worker_thread = WorkerThread(encryption_key, playlist, mode)
            self.worker_thread.progress_changed.connect(self.update_progress_bar)
            self.worker_thread.finished.connect(self.on_task_finished)
            self.worker_thread.start()
        except Exception as e:
            self.status_label.setHidden(False)
            self.status_label.setText(str(e))
            print(e)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def on_task_finished(self):
        time.sleep(1)
        self.progress_bar.setHidden(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AutocuepointsGUI()
    app.exec()
