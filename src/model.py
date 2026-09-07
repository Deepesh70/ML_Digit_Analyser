"""
model.py
--------
Defines the Convolutional Neural Network (CNN) architecture for
handwritten digit classification (MNIST).
"""

from tensorflow.keras import layers, models


def build_digit_model() -> models.Sequential:
    """
    Builds and returns a CNN model optimized for 28x28 grayscale digit images.

    Architecture summary:
    - Input: (28, 28, 1)
    - Conv Block 1: 32 filters (3x3), ReLU, MaxPool (2x2), Dropout (0.25)
    - Conv Block 2: 64 filters (3x3), ReLU, MaxPool (2x2), Dropout (0.25)
    - Flatten: Convert 2D feature maps to 1D vector
    - Fully Connected: 128 units, ReLU, Dropout (0.50)
    - Output: 10 units, Softmax (probabilities for digits 0-9)
    """
    model = models.Sequential([
        # 1. Explicit Input layer
        layers.Input(shape=(28, 28, 1)),

        # 2. First Convolutional Block (Low-level features: edges, curves)
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # 3. Second Convolutional Block (Mid-level features: loops, angles)
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # 4. Dense Classifier
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.50),

        # 5. Output Layer (10 classes: 0 through 9)
        layers.Dense(10, activation="softmax")
    ])

    return model


if __name__ == "__main__":
    # Quick sanity check: build model and print architecture summary
    model = build_digit_model()
    model.summary()
