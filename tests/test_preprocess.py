import numpy as np
from src.preprocess import preprocess_digit_image


def test_prepare_image_shape():
    """Ensure raw canvas array is processed into (1, 28, 28, 1) float32 and (28, 28) preview."""
    # Simulate a 280x280 RGBA canvas output from streamlit-drawable-canvas
    dummy_rgba = np.zeros((280, 280, 4), dtype=np.uint8)
    # Draw a mock white square in center
    dummy_rgba[100:180, 100:180, :3] = 255
    dummy_rgba[100:180, 100:180, 3] = 255

    tensor, preview = preprocess_digit_image(dummy_rgba)

    assert tensor.shape == (1, 28, 28, 1)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    assert preview.shape == (28, 28)

