from track_interface.track_interface import TrackInterface
from pyrekordbox import Rekordbox6Database
import librosa
import librosa.display
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files


SAMPLE_RATE = 1000
HOP_LENGTH = 50


def get_recurrence_matrix(track_filepath: str):
    """Compute MFCC and recurrence matrix for a track"""
    print(f"Computing MFCC and recurrence matrix for {os.path.basename(track_filepath)}")

    # Load audio file
    y, sr = librosa.load(track_filepath, sr=SAMPLE_RATE)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)

    # Compute recurrence matrix using MFCC features
    recurrence_matrix = librosa.segment.recurrence_matrix(mfcc,
                                                          metric='cosine',
                                                          mode='affinity',
                                                          sym=True)
    print(f"Recurrence matrix has size {recurrence_matrix.shape}")

    return recurrence_matrix


def load_rekordbox_cuepoints(track_filepath: str):
    """Load cuepoints from Rekordbox database for a given track"""
    try:
        db = Rekordbox6Database()
        playlist = db.get_playlist(Name="training_data").one()

        for song in playlist.Songs:
            ti = TrackInterface(song, db)
            if ti.get_content_filepath() == track_filepath:
                # Convert from milliseconds to seconds
                cuepoints = [cue / 1000.0 for cue in ti.read_hot_cues()]
                print(f"  Found {len(cuepoints)} cuepoints")
                return cuepoints

        print(f"  No Rekordbox data found for this track")
        return []

    except Exception as e:
        print(f"  Error loading cuepoints: {str(e)}")
        return []


def save_recurrence_matrix_as_image(recurrence_matrix, output_path, track_name, cuepoints=None):
    """Save recurrence matrix as an image file with optional cuepoints"""

    fig = plt.figure(figsize=(12, 10))
    librosa.display.specshow(recurrence_matrix,
                             x_axis='time',
                             y_axis='time',
                             cmap='hot',
                             sr=SAMPLE_RATE,
                             hop_length=HOP_LENGTH)
    plt.colorbar(label='Affinity')

    # Add cuepoints if available
    if cuepoints:
        for cuepoint in cuepoints:
            plt.axvline(x=cuepoint, color='blue', linestyle='--', linewidth=2, alpha=0.7)
            plt.axhline(y=cuepoint, color='blue', linestyle='--', linewidth=2, alpha=0.7)

        title = f'Recurrence Matrix - {track_name} ({len(cuepoints)} cuepoints)'
    else:
        title = f'Recurrence Matrix - {track_name}'

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Time (s)')
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_folder(input_folder, output_folder, load_cuepoints=False):
    """Process all audio files in a folder and save recurrence matrices as images

    Args:
        input_folder: Path to folder containing audio files
        output_folder: Path to folder where images will be saved
        load_cuepoints: If True, load cuepoints from Rekordbox and overlay them on the images
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Supported audio formats
    audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']

    # Get all audio files in the folder
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(input_folder).glob(f'*{ext}'))

    mode = "labeled" if load_cuepoints else "unlabeled"
    print(f"Found {len(audio_files)} audio files to process ({mode} mode)\n")

    # Process each file
    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")

            # Compute recurrence matrix
            recurrence_matrix = get_recurrence_matrix(str(audio_file))

            # Load cuepoints if requested
            cuepoints = []
            if load_cuepoints:
                cuepoints = load_rekordbox_cuepoints(str(audio_file))

            # Create output filename (same name but with .png extension)
            output_filename = audio_file.stem + '_recurrence.png'
            output_path = os.path.join(output_folder, output_filename)

            # Save recurrence matrix as image with cuepoints
            save_recurrence_matrix_as_image(recurrence_matrix, output_path, audio_file.stem, cuepoints)
            print(f"✓ Saved to: {output_path}\n")

        except Exception as e:
            print(f"✗ Error processing {audio_file.name}: {str(e)}\n")
            continue

    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Saved {len(list(Path(output_folder).glob('*.png')))} recurrence matrix images to: {output_folder}")
    print(f"{'='*50}")


if __name__ == "__main__":
    # Configure your paths here
    input_folder = "/Users/shivamenta/Desktop/training_data_stems"

    # Example 1: Generate unlabeled recurrence matrices (no cuepoints)
    output_folder_unlabeled = "/Users/shivamenta/Desktop/stems_recurrence_matrices"
    process_folder(input_folder, output_folder_unlabeled, load_cuepoints=False)

