import os
from pathlib import Path

import librosa
import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
import torch
import torchaudio
from audio_separator.separator import Separator


# TODO(smenta) – Understand the internals of what this is actually doing.
def split_to_vocals_and_instrumentals(input_path, output_path):
    separator = Separator(output_dir=output_path)
    separator.load_model()  # use mdx model
    output_files = separator.separate(input_path)
    print(f"Separation complete! Output file(s): {' '.join(output_files)}")


def process_folder(input_folder, output_folder):
    """Process all audio files in a folder and split stems using MDX model."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    split_to_vocals_and_instrumentals(input_folder, output_folder)
    print(f"Processing complete!")


if __name__ == "__main__":
    # Configure your paths here
    input_folder = "/Users/shivamenta/Desktop/training_data/"
    output_folder = "/Users/shivamenta/Desktop/training_data_stems/"

    # Process all songs
    process_folder(input_folder, output_folder)
