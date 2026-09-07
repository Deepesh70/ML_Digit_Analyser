"""
preprocess.py
-------------
Transforms real-world images (uploaded files, photos, or canvas drawings)
into the exact format expected by the MNIST-trained CNN:
1. Grayscale conversion
2. Contrast/inversion detection (white digit on black background)
3. Bounding-box cropping
4. Aspect-ratio-preserving resize into a 20x20 box
5. Center-of-mass centering inside a 28x28 canvas
6. Pixel normalization to [0.0, 1.0]
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass


def preprocess_digit_image(
    image_input: Union[str, Image.Image, np.ndarray],
    threshold: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses any input image into the MNIST standard format.

    Args:
        image_input: Filepath (str), PIL Image, or NumPy array.
        threshold: Pixel intensity threshold to separate ink from background.

    Returns:
        tensor: Shape (1, 28, 28, 1), float32 normalized to [0.0, 1.0], ready for model.predict().
        preview_28x28: Shape (28, 28), uint8 image (0-255) for UI visualization.
    """
    # 1. Convert any input format into a PIL Grayscale Image
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, np.ndarray):
        # Handle RGBA from canvas or RGB arrays
        if image_input.ndim == 3 and image_input.shape[2] == 4:
            rgb = image_input[:, :, :3]
            if np.any(rgb > 0):
                # Standard RGB canvas drawing
                img = Image.fromarray(rgb).convert("L")
            else:
                # Fallback: transparent canvas with alpha stroke mask
                img = Image.fromarray(image_input[:, :, 3]).convert("L")
        elif image_input.ndim == 3:
            img = Image.fromarray(image_input).convert("L")
        else:
            img = Image.fromarray(image_input.astype(np.uint8)).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    arr = np.array(img, dtype=np.float32)

    # 2. Inversion check: MNIST is white ink (255) on black background (0).
    # If the corners/edges are bright (paper/white canvas), invert the image.
    corner_samples = np.concatenate([
        arr[:5, :5].flatten(),
        arr[:5, -5:].flatten(),
        arr[-5:, :5].flatten(),
        arr[-5:, -5:].flatten()
    ])
    if np.mean(corner_samples) > 127:
        arr = 255.0 - arr

    # Clean up faint noise
    arr[arr < threshold] = 0.0

    # 3. Find bounding box of the drawn digit
    rows = np.any(arr > threshold, axis=1)
    cols = np.any(arr > threshold, axis=0)

    if not np.any(rows) or not np.any(cols):
        # Blank image submitted
        blank = np.zeros((28, 28), dtype=np.float32)
        tensor = blank.reshape(1, 28, 28, 1)
        return tensor, np.zeros((28, 28), dtype=np.uint8)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop tightly to the digit
    digit_crop = arr[rmin:rmax + 1, cmin:cmax + 1]

    # 4. Resize to fit inside a 20x20 box preserving aspect ratio
    crop_h, crop_w = digit_crop.shape
    crop_img = Image.fromarray(digit_crop)

    if crop_h > crop_w:
        factor = 20.0 / crop_h
        new_h = 20
        new_w = max(1, int(round(crop_w * factor)))
    else:
        factor = 20.0 / crop_w
        new_w = 20
        new_h = max(1, int(round(crop_h * factor)))

    resized = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_arr = np.array(resized, dtype=np.float32)

    # 5. Place the 20x20 (or smaller) digit into a 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_arr

    # 6. Center by Center of Mass (Standard MNIST convention)
    cy, cx = center_of_mass(canvas)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_y = int(round(14.0 - cy))
        shift_x = int(round(14.0 - cx))
        canvas = np.roll(canvas, shift_y, axis=0)
        canvas = np.roll(canvas, shift_x, axis=1)

    # 7. Normalize pixel range to [0.0, 1.0] and format tensor
    canvas = np.clip(canvas, 0.0, 255.0)
    preview_28x28 = canvas.astype(np.uint8)
    tensor = (canvas / 255.0).reshape(1, 28, 28, 1).astype(np.float32)

    return tensor, preview_28x28


if __name__ == "__main__":
    # Self-test with a synthetic dummy array
    test_drawing = np.zeros((200, 200), dtype=np.uint8)
    test_drawing[50:150, 90:110] = 255  # Vertical stroke representing a "1"
    tensor, preview = preprocess_digit_image(test_drawing)
    print("Preprocessing successful!")
    print(f"Tensor shape: {tensor.shape}, min: {tensor.min()}, max: {tensor.max()}")
    print(f"Preview image shape: {preview.shape}")
