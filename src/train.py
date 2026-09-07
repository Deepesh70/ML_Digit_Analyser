"""
train.py
--------
Training pipeline for handwritten digit recognition using the EMNIST (Extended MNIST) Digits dataset.
Features:
- Fixes upstream NIST download URL to https://biometrics.nist.gov
- Loads EMNIST Digits (240,000 train + 40,000 test samples)
- Normalization to [0.0, 1.0] and channel dimension formatting
- Real-time data augmentation (rotation, translation, zoom)
- Modern training callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
- Exports high-accuracy model to models/best_digit_model.keras
"""

import os
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path so 'src' can be imported from anywhere
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf
from tensorflow.keras import callbacks, layers
from src.model import build_digit_model


def load_and_preprocess_emnist():
    """
    Loads EMNIST Digits dataset with patched active NIST download server.
    Cached location: ~/.cache/emnist/emnist.zip (~535 MB)
    EMNIST Digits split:
        Train: 240,000 images
        Test:   40,000 images
    """
    print("[1/4] Loading EMNIST (Extended MNIST - Digits) dataset...")
    try:
        import emnist
    except ImportError:
        print("\n[ERROR] 'emnist' library is not installed.")
        print("Please run in your terminal: pip install emnist\n")
        sys.exit(1)

    # Configure dataset to be stored directly inside this repository under data/
    data_dir = ROOT_DIR / "data"
    os.makedirs(data_dir, exist_ok=True)
    emnist.CACHE_FILE_PATH = str(data_dir / "emnist.zip")

    # CRITICAL FIX: NIST changed their official download domain from itl.nist.gov to biometrics.nist.gov.
    # The old URL returns an HTML error page, causing a 'BadZipFile' error.
    # We patch the source URL directly to the active NIST server:
    emnist.SOURCE_URLS = [
        "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    ]

    print(f"      Dataset path: {emnist.CACHE_FILE_PATH}")
    print("      (Downloading official ~535 MB archive directly into ./data/...)")

    x_train, y_train = emnist.extract_training_samples("digits")
    x_test, y_test = emnist.extract_test_samples("digits")

    # Normalize pixel values to 0.0 - 1.0
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    print(f"      Training set: {x_train.shape[0]:,} images (28x28 grayscale)")
    print(f"      Test set:     {x_test.shape[0]:,} images")
    return (x_train, y_train), (x_test, y_test)


def create_data_augmentation():
    """
    Data Augmentation pipeline.
    Simulates real-world drawing variations on the fly.
    """
    return tf.keras.Sequential([
        layers.RandomRotation(0.06),          # Slight rotation
        layers.RandomTranslation(0.06, 0.06), # Slight translation
        layers.RandomZoom(0.06)               # Slight zoom
    ], name="data_augmentation")


def train(epochs: int = 10, batch_size: int = 128):
    """
    Runs the full model training and evaluation pipeline on EMNIST Digits.
    """
    # Create models directory if it doesn't exist
    os.makedirs(ROOT_DIR / "models", exist_ok=True)
    model_save_path = str(ROOT_DIR / "models" / "best_digit_model.keras")

    # 1. Load EMNIST dataset
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_emnist()

    # 2. Build model architecture
    print("[2/4] Building CNN model...")
    base_model = build_digit_model()

    # Prepend data augmentation to the model
    augmentation = create_data_augmentation()
    inputs = layers.Input(shape=(28, 28, 1))
    augmented = augmentation(inputs)
    outputs = base_model(augmented)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="augmented_emnist_cnn")

    # 3. Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Callbacks
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 5. Train
    print(f"[3/4] Training on 240,000 EMNIST samples for up to {epochs} epochs (Batch size: {batch_size})...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,  # 24,000 validation images
        callbacks=training_callbacks,
        verbose=1
    )

    # 6. Evaluate on unseen test set
    print("\n[4/4] Evaluating best model on 40,000 unseen EMNIST test images...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("=" * 55)
    print(f" EMNIST Final Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f" EMNIST Final Test Loss     : {test_loss:.4f}")
    print(f" Model successfully saved to: {model_save_path}")
    print("=" * 55)

    return model, history


if __name__ == "__main__":
    train()
