"""
Script to train the ML-based cuepoint detection engine.
Trains on labeled training data and saves the model.
"""

import pickle
import warnings
from typing import List, Tuple

import numpy as np
from pyrekordbox import Rekordbox6Database
from tqdm import tqdm

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Please install required packages: pip install xgboost scikit-learn")
    exit(1)

from track_interface.cuepoint_engines.ml_engine import MLEngine
from track_interface.track_interface import TrackInterface

warnings.filterwarnings("ignore", category=DeprecationWarning)


def extract_training_data(db, playlist_name: str = "training_data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels from training data.
    Returns: (X, y) where X is features and y is binary labels (1=cuepoint, 0=not)
    """
    playlist = db.get_playlist(Name=playlist_name).one()

    all_features = []
    all_labels = []

    print(f"Extracting features from {len(playlist.Songs)} songs...")
    ml_engine = MLEngine()

    for song in tqdm(playlist.Songs):
        try:
            # Create TrackInterface to properly access song data
            ti = TrackInterface(song, db)

            # Extract features for all measures
            X_song, measure_timestamps = ml_engine.extract_all_features()

            if len(X_song) == 0:
                continue

            # Get actual cuepoints from the song
            actual_cuepoints = ti.read_hot_cues()

            # Create labels: 1 if measure has a cuepoint, 0 otherwise
            # Use a tolerance window (e.g., within 500ms)
            tolerance_ms = 500

            y_song = []
            for timestamp in measure_timestamps:
                is_cuepoint = any(
                    abs(timestamp - cp) <= tolerance_ms
                    for cp in actual_cuepoints
                )
                y_song.append(1 if is_cuepoint else 0)

            all_features.append(X_song)
            all_labels.extend(y_song)

        except Exception as e:
            print(f"Error processing song: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Concatenate all features
    X = np.vstack(all_features)
    y = np.array(all_labels)

    print(f"\nDataset statistics:")
    print(f"Total measures: {len(y)}")
    print(f"Positive samples (cuepoints): {np.sum(y)} ({100*np.mean(y):.2f}%)")
    print(f"Negative samples: {len(y) - np.sum(y)} ({100*(1-np.mean(y)):.2f}%)")
    print(f"Feature dimensions: {X.shape[1]}")

    return X, y


def train_model(X: np.ndarray, y: np.ndarray, model_path: str = "ml_cuepoint_model.pkl"):
    """
    Train XGBoost classifier on the extracted features.
    """
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Handle class imbalance with scale_pos_weight
    scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # Train XGBoost model
    print("\nTraining XGBoost classifier...")

    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20,
        tree_method='hist'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Evaluate on validation set
    print("\nValidation set performance:")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    print(classification_report(y_val, y_pred, target_names=['Not Cuepoint', 'Cuepoint']))

    print(f"\nPrecision: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")

    # Feature importance
    print("\nTop 10 most important features:")
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for idx in top_indices:
        print(f"Feature {idx}: {feature_importance[idx]:.4f}")

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\nModel saved to {model_path}")

    return model


def main():
    if not SKLEARN_AVAILABLE:
        print("Error: Required packages not installed.")
        print("Install with: pip install xgboost scikit-learn")
        return

    # Connect to Rekordbox database
    db = Rekordbox6Database()

    # Extract training data
    X, y = extract_training_data(db, playlist_name="training_data")

    # Train model
    model = train_model(X, y, model_path="ml_cuepoint_model.pkl")

    print("\nTraining complete!")
    print("You can now use MLEngine to generate cuepoints with the trained model.")


if __name__ == "__main__":
    main()
