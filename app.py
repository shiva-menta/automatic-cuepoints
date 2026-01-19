from typing import List
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QVBoxLayout,
    QWidget,
    QToolButton,
    QSizePolicy,
)
from PyQt5.QtGui import QPainter, QColor, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
import sys
import multiprocessing
from enum import Enum

# Get the directory where this script is located (handles PyInstaller bundling)
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = sys._MEIPASS
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODAL_ICON_PATH = os.path.join(SCRIPT_DIR, "Modal-IconMark.png")


# =============================================================================
# Enums
# =============================================================================

class DeploymentStatus(Enum):
    IDLE = "idle"
    SUCCESS = "success"
    FAILED = "failed"


class ModelType(Enum):
    LOCAL = "local"
    REMOTE = "remote"


# =============================================================================
# Stub Functions - Fill these out
# =============================================================================

def validate_modal_credentials(token_id: str, token_secret: str) -> bool:
    """
    Validate Modal credentials.

    Args:
        token_id: The Modal Token ID
        token_secret: The Modal Token Secret

    Returns:
        True if credentials are valid (non-empty), False otherwise
    """
    return bool(token_id and token_secret)


def fetch_rekordbox_playlists(encryption_key: str) -> List[str]:
    """
    Fetch available playlists from Rekordbox database.

    Args:
        encryption_key: The Rekordbox database encryption key (can be empty)

    Returns:
        List of playlist names
    """
    from pyrekordbox import Rekordbox6Database

    try:
        if encryption_key:
            db = Rekordbox6Database(key=encryption_key)
        else:
            db = Rekordbox6Database()

        playlists = db.get_playlist()
        return [playlist.Name for playlist in playlists if playlist.Name]
    except Exception:
        return []


def process_cuepoints_add(encryption_key: str, playlist: str, model_type: ModelType, progress_callback) -> None:
    """
    Add cuepoints to all tracks in the playlist.

    Args:
        encryption_key: The Rekordbox database encryption key
        playlist: The playlist name to process
        model_type: LOCAL or REMOTE model
        progress_callback: Function to call with progress (0-100)
    """
    from cuepoint_utils import CuepointProcessingArgs, add_cuepoints_to_playlist

    # Select model based on type
    model = "all_in_one" if model_type == ModelType.REMOTE else "stft"

    args = CuepointProcessingArgs(
        model=model,
        num_processes=4,
        encryption_key=encryption_key if encryption_key else None,
    )

    # Adapt (current, total) callback to percentage callback for GUI
    def adapted_callback(current: int, total: int):
        if progress_callback and total > 0:
            progress_callback(int(current / total * 100))

    add_cuepoints_to_playlist(
        playlist_name=playlist,
        args=args,
        progress_callback=adapted_callback,
    )


def process_cuepoints_clear(encryption_key: str, playlist: str, progress_callback) -> None:
    """
    Clear cuepoints from all tracks in the playlist.

    Args:
        encryption_key: The Rekordbox database encryption key
        playlist: The playlist name to process
        progress_callback: Function to call with progress (0-100)
    """
    from cuepoint_utils import clear_cuepoints_from_playlist

    # Adapt (current, total) callback to percentage callback for GUI
    def adapted_callback(current: int, total: int):
        if progress_callback and total > 0:
            progress_callback(int(current / total * 100))

    clear_cuepoints_from_playlist(
        playlist_name=playlist,
        encryption_key=encryption_key if encryption_key else None,
        progress_callback=adapted_callback,
    )


# =============================================================================
# Custom Widgets
# =============================================================================

class StatusIndicator(QWidget):
    """A colored circle indicator for deployment status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._status = DeploymentStatus.IDLE
        self.setFixedSize(16, 16)

    def set_status(self, status: DeploymentStatus):
        self._status = status
        self.update()

    def get_status(self) -> DeploymentStatus:
        return self._status

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        colors = {
            DeploymentStatus.IDLE: QColor(128, 128, 128),      # Gray
            DeploymentStatus.SUCCESS: QColor(76, 175, 80),     # Green
            DeploymentStatus.FAILED: QColor(244, 67, 54),      # Red
        }

        painter.setBrush(colors[self._status])
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, 12, 12)


class CollapsibleSection(QWidget):
    """A collapsible section widget with header and content."""

    def __init__(self, title: str, parent=None, header_widget: QWidget = None, icon_path: str = None):
        super().__init__(parent)

        self._is_expanded = False

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Header row (button + optional widget)
        self.header_row = QWidget()
        self.header_row.setStyleSheet("background-color: #e0e0e0; border-radius: 4px;")
        header_row_layout = QHBoxLayout(self.header_row)
        header_row_layout.setContentsMargins(0, 0, 8, 0)
        header_row_layout.setSpacing(4)

        # Header button
        self.header_button = QToolButton()
        self.header_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header_button.setArrowType(Qt.RightArrow)
        self.header_button.setText(title)
        self.header_button.setCheckable(True)
        self.header_button.setChecked(False)
        self.header_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header_button.setStyleSheet("""
            QToolButton {
                border: none;
                padding: 8px;
                font-weight: bold;
                font-size: 13px;
                text-align: left;
                background-color: #e0e0e0;
                color: #212121;
                border-radius: 4px;
            }
            QToolButton:hover {
                background-color: #d0d0d0;
            }
        """)
        self.header_button.clicked.connect(self._toggle)

        header_row_layout.addWidget(self.header_button)

        # Optional icon between title and header widget
        if icon_path and os.path.exists(icon_path):
            icon_label = QLabel()
            pixmap = QPixmap(icon_path)
            if not pixmap.isNull():
                icon_label.setPixmap(pixmap.scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            icon_label.setFixedSize(24, 24)
            icon_label.setStyleSheet("background-color: transparent;")
            header_row_layout.addWidget(icon_label)

        if header_widget:
            header_row_layout.addWidget(header_widget)

        # Content widget
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background-color: #fafafa;")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_widget.setVisible(False)

        self.main_layout.addWidget(self.header_row)
        self.main_layout.addWidget(self.content_widget)

    def _toggle(self):
        self._is_expanded = not self._is_expanded
        self.header_button.setArrowType(Qt.DownArrow if self._is_expanded else Qt.RightArrow)
        self.content_widget.setVisible(self._is_expanded)

    def set_expanded(self, expanded: bool):
        if self._is_expanded != expanded:
            self._toggle()
            self.header_button.setChecked(expanded)

    def add_widget(self, widget: QWidget):
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)


# =============================================================================
# Worker Threads
# =============================================================================

class CuepointWorker(QThread):
    """Worker thread for cuepoint processing."""
    progress_changed = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, encryption_key: str, playlist: str, mode: str, model_type: ModelType):
        super().__init__()
        self.encryption_key = encryption_key
        self.playlist = playlist
        self.mode = mode
        self.model_type = model_type

    def run(self):
        try:
            if self.mode == "add":
                process_cuepoints_add(
                    self.encryption_key,
                    self.playlist,
                    self.model_type,
                    self.progress_changed.emit
                )
            else:
                process_cuepoints_clear(
                    self.encryption_key,
                    self.playlist,
                    self.progress_changed.emit
                )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# Main Application Window
# =============================================================================

class AutocuepointsGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Autocuepoints")
        self.setMinimumWidth(450)

        self._deployment_status = DeploymentStatus.IDLE
        self._selected_model = ModelType.LOCAL

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # Set hardcoded colors for the main window
        self.setStyleSheet("""
            QWidget {
                background-color: #fafafa;
                color: #212121;
            }
            QLineEdit {
                background-color: #ffffff;
                color: #212121;
                border: 1px solid #bdbdbd;
                border-radius: 4px;
                padding: 6px;
            }
            QLineEdit:focus {
                border: 1px solid #1976d2;
            }
            QComboBox {
                background-color: #ffffff;
                color: #212121;
                border: 1px solid #bdbdbd;
                border-radius: 4px;
                padding: 6px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #212121;
                selection-background-color: #e3f2fd;
                selection-color: #212121;
            }
            QPushButton {
                background-color: #1976d2;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #bdbdbd;
                color: #757575;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: #ffffff;
                color: #212121;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #212121;
            }
            QRadioButton {
                color: #212121;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: #e0e0e0;
                text-align: center;
                color: #212121;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 3px;
            }
            QLabel {
                color: #212121;
            }
        """)

        # Modal Section (collapsed by default) - with icon and status indicator in header
        self.status_indicator = StatusIndicator()
        self.modal_section = CollapsibleSection(
            "Modal",
            header_widget=self.status_indicator,
            icon_path=MODAL_ICON_PATH
        )
        self._setup_modal_section()
        main_layout.addWidget(self.modal_section)

        # Cuepoints Section (expanded by default)
        self.cuepoints_section = CollapsibleSection("Cuepoints")
        self._setup_cuepoints_section()
        self.cuepoints_section.set_expanded(True)
        main_layout.addWidget(self.cuepoints_section)

        # Stretch to push content up
        main_layout.addStretch()

        self.show()

    def _setup_modal_section(self):
        # Token ID
        token_id_label = QLabel("Token ID:")
        self.token_id_input = QLineEdit()
        self.token_id_input.setPlaceholderText("Enter Modal Token ID...")
        self.token_id_input.textChanged.connect(self._on_modal_credentials_changed)

        # Token Secret
        token_secret_label = QLabel("Token Secret:")
        self.token_secret_input = QLineEdit()
        self.token_secret_input.setEchoMode(QLineEdit.Password)
        self.token_secret_input.setPlaceholderText("Enter Modal Token Secret...")
        self.token_secret_input.textChanged.connect(self._on_modal_credentials_changed)

        self.modal_section.add_widget(token_id_label)
        self.modal_section.add_widget(self.token_id_input)
        self.modal_section.add_widget(token_secret_label)
        self.modal_section.add_widget(self.token_secret_input)

    def _setup_cuepoints_section(self):
        # Rekordbox GroupBox
        rekordbox_group = QGroupBox("Rekordbox")
        rekordbox_layout = QVBoxLayout(rekordbox_group)

        # Encryption Key
        encryption_label = QLabel("Encryption Key:")
        self.encryption_input = QLineEdit()
        self.encryption_input.setPlaceholderText("Leave empty if not required")

        # Playlist dropdown
        playlist_label = QLabel("Playlist:")
        self.playlist_dropdown = QComboBox()
        self.playlist_dropdown.addItem("Select playlist...")

        # Refresh playlists button
        playlist_row = QHBoxLayout()
        playlist_row.addWidget(self.playlist_dropdown, stretch=1)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setFixedWidth(90)
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #bdbdbd;
                color: #757575;
            }
        """)
        self.refresh_button.clicked.connect(self._on_refresh_playlists)
        playlist_row.addWidget(self.refresh_button)

        # Disclaimer
        disclaimer_label = QLabel("⚠ Please close Rekordbox before proceeding")
        disclaimer_label.setStyleSheet("color: #ff9800; font-size: 11px; padding: 4px;")

        rekordbox_layout.addWidget(encryption_label)
        rekordbox_layout.addWidget(self.encryption_input)
        rekordbox_layout.addWidget(playlist_label)
        rekordbox_layout.addLayout(playlist_row)
        rekordbox_layout.addWidget(disclaimer_label)

        self.cuepoints_section.add_widget(rekordbox_group)

        # Model Selection
        model_label = QLabel("Model Selection")
        model_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        self.cuepoints_section.add_widget(model_label)

        model_frame = QFrame()
        model_frame.setFrameStyle(QFrame.StyledPanel)
        model_frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 4px;
            }
            QWidget {
                background-color: #f5f5f5;
            }
        """)
        model_layout = QVBoxLayout(model_frame)
        model_layout.setSpacing(8)

        self.model_button_group = QButtonGroup(self)

        # Local option
        local_widget = QWidget()
        local_layout = QHBoxLayout(local_widget)
        local_layout.setContentsMargins(0, 0, 0, 0)

        self.local_radio = QRadioButton()
        self.local_radio.setChecked(True)
        self.model_button_group.addButton(self.local_radio, 0)

        local_name_label = QLabel("🏠 Local")
        local_name_label.setFixedWidth(80)
        local_speed_label = QLabel("⚡ Fast")
        local_speed_label.setStyleSheet("color: #4caf50;")
        local_accuracy_label = QLabel("🎯 Lower Accuracy")
        local_accuracy_label.setStyleSheet("color: #ff9800;")

        local_layout.addWidget(self.local_radio)
        local_layout.addWidget(local_name_label)
        local_layout.addWidget(local_speed_label)
        local_layout.addWidget(local_accuracy_label)
        local_layout.addStretch()

        # Remote option
        remote_widget = QWidget()
        remote_layout = QHBoxLayout(remote_widget)
        remote_layout.setContentsMargins(0, 0, 0, 0)

        self.remote_radio = QRadioButton()
        self.remote_radio.setEnabled(False)  # Disabled until deployment succeeds
        self.model_button_group.addButton(self.remote_radio, 1)

        # Remote name with Modal icon inline (using rich text)
        remote_name_label = QLabel()
        if os.path.exists(MODAL_ICON_PATH):
            # Use HTML with embedded image for inline icon (Modal logo is ~2:1 aspect ratio)
            remote_name_label.setText(f'<img src="{MODAL_ICON_PATH}" width="28" height="14"> Remote')
        else:
            remote_name_label.setText("☁️ Remote")
        remote_name_label.setFixedWidth(105)

        remote_speed_label = QLabel("⚡ Fast")
        remote_speed_label.setStyleSheet("color: #4caf50;")
        remote_accuracy_label = QLabel("🎯 Better Accuracy")
        remote_accuracy_label.setStyleSheet("color: #4caf50;")

        remote_layout.addWidget(self.remote_radio)
        remote_layout.addWidget(remote_name_label)
        remote_layout.addWidget(remote_speed_label)
        remote_layout.addWidget(remote_accuracy_label)
        remote_layout.addStretch()

        # Remote disabled notice (warning style like Rekordbox disclaimer)
        self.remote_notice = QLabel("⚠ Enter Modal credentials first")
        self.remote_notice.setStyleSheet("color: #ff9800; font-size: 11px; padding: 4px; margin-left: 20px;")

        model_layout.addWidget(local_widget)
        model_layout.addWidget(remote_widget)
        model_layout.addWidget(self.remote_notice)

        self.cuepoints_section.add_widget(model_frame)

        # Action buttons
        buttons_layout = QHBoxLayout()

        button_style = """
            QPushButton {
                background-color: #1976d2;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #bdbdbd;
                color: #757575;
            }
        """

        self.add_cuepoints_button = QPushButton("Add Cuepoints")
        self.add_cuepoints_button.setStyleSheet(button_style)
        self.add_cuepoints_button.clicked.connect(self._on_add_cuepoints)

        self.clear_cuepoints_button = QPushButton("Clear Cuepoints")
        self.clear_cuepoints_button.setStyleSheet(button_style)
        self.clear_cuepoints_button.clicked.connect(self._on_clear_cuepoints)

        buttons_layout.addWidget(self.add_cuepoints_button)
        buttons_layout.addWidget(self.clear_cuepoints_button)

        self.cuepoints_section.add_layout(buttons_layout)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.cuepoints_section.add_widget(self.progress_bar)

        # Status label for errors
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #f44336;")
        self.status_label.setVisible(False)
        self.cuepoints_section.add_widget(self.status_label)

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_modal_credentials_changed(self):
        token_id = self.token_id_input.text()
        token_secret = self.token_secret_input.text()

        if validate_modal_credentials(token_id, token_secret):
            self._deployment_status = DeploymentStatus.SUCCESS
            self.status_indicator.set_status(DeploymentStatus.SUCCESS)
            self.remote_radio.setEnabled(True)
            self.remote_notice.setVisible(False)

            os.environ["MODAL_TOKEN_ID"] = token_id
            os.environ["MODAL_TOKEN_SECRET"] = token_secret
        else:
            self._deployment_status = DeploymentStatus.IDLE
            self.status_indicator.set_status(DeploymentStatus.IDLE)
            self.remote_radio.setEnabled(False)
            self.remote_notice.setText("⚠ Enter Modal credentials first")
            self.remote_notice.setVisible(True)
            # If remote was selected, switch back to local
            if self.remote_radio.isChecked():
                self.local_radio.setChecked(True)

    def _on_refresh_playlists(self):
        encryption_key = self.encryption_input.text()

        self.playlist_dropdown.clear()
        self.playlist_dropdown.addItem("Loading...")

        playlists = fetch_rekordbox_playlists(encryption_key)

        self.playlist_dropdown.clear()
        if playlists:
            for playlist in playlists:
                self.playlist_dropdown.addItem(playlist)
        else:
            self.playlist_dropdown.addItem("No playlists found")

    def _on_add_cuepoints(self):
        self._run_cuepoint_operation("add")

    def _on_clear_cuepoints(self):
        self._run_cuepoint_operation("clear")

    def _run_cuepoint_operation(self, mode: str):
        playlist = self.playlist_dropdown.currentText()
        if playlist in ["Select playlist...", "Loading...", "No playlists found"]:
            self.status_label.setText("Please select a valid playlist")
            self.status_label.setVisible(True)
            return

        encryption_key = self.encryption_input.text()
        model_type = ModelType.REMOTE if self.remote_radio.isChecked() else ModelType.LOCAL

        # Disable buttons and show progress
        self.add_cuepoints_button.setEnabled(False)
        self.clear_cuepoints_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(False)

        self.cuepoint_worker = CuepointWorker(encryption_key, playlist, mode, model_type)
        self.cuepoint_worker.progress_changed.connect(self._on_progress_changed)
        self.cuepoint_worker.finished.connect(self._on_cuepoint_operation_finished)
        self.cuepoint_worker.error.connect(self._on_cuepoint_operation_error)
        self.cuepoint_worker.start()

    def _on_progress_changed(self, value: int):
        self.progress_bar.setValue(value)

    def _on_cuepoint_operation_finished(self):
        self.add_cuepoints_button.setEnabled(True)
        self.clear_cuepoints_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_cuepoint_operation_error(self, error_message: str):
        self.add_cuepoints_button.setEnabled(True)
        self.clear_cuepoints_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(error_message)
        self.status_label.setVisible(True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = AutocuepointsGUI()
    sys.exit(app.exec())
