"""
analyser.py
-----------
Advanced diagnostic and geometric analysis engine ("The Brain of ML Digit Analyser"):
1. Analyzes stroke morphology (aspect ratio, density, bounding box)
2. Detects tilted, sideways, or ambiguous digit drawings
3. Runs multi-angle rotation analysis (0°, 90° CW, 180°, 90° CCW)
4. Produces human-interpretable engineering advisories
"""

from typing import Dict, Any, List
import numpy as np
from PIL import Image
from src.preprocess import preprocess_digit_image


def analyze_stroke_geometry(raw_drawing: np.ndarray, threshold: int = 40) -> Dict[str, Any]:
    """
    Computes geometric metrics of the drawn stroke.

    Returns:
        width: Bounding box width
        height: Bounding box height
        aspect_ratio: width / height
        is_unusual_aspect: True if stroke is unnaturally wide or unnaturally tall
        stroke_pixel_count: Number of active ink pixels
    """
    if raw_drawing.ndim == 3:
        if raw_drawing.shape[2] == 4:
            # Check RGB channels
            rgb = raw_drawing[:, :, :3]
            if np.any(rgb > 0):
                gray = np.array(Image.fromarray(rgb).convert("L"))
            else:
                gray = raw_drawing[:, :, 3]
        else:
            gray = np.array(Image.fromarray(raw_drawing).convert("L"))
    else:
        gray = raw_drawing

    rows = np.any(gray > threshold, axis=1)
    cols = np.any(gray > threshold, axis=0)

    if not np.any(rows) or not np.any(cols):
        return {
            "is_empty": True,
            "width": 0,
            "height": 0,
            "aspect_ratio": 1.0,
            "is_unusual_aspect": False,
            "stroke_pixel_count": 0,
        }

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h = int(rmax - rmin + 1)
    w = int(cmax - cmin + 1)
    aspect = float(w / max(1, h))
    pixel_count = int(np.sum(gray > threshold))

    # Standard handwritten digits are taller than wide (aspect between 0.35 and 1.2)
    # If aspect > 1.6, the digit is more than 60% wider than tall (unusual for standard upright digits)
    is_unusual = aspect > 1.55 or aspect < 0.22

    return {
        "is_empty": False,
        "width": w,
        "height": h,
        "aspect_ratio": round(aspect, 2),
        "is_unusual_aspect": is_unusual,
        "stroke_pixel_count": pixel_count,
    }


def analyze_multi_angle_predictions(
    raw_drawing: np.ndarray,
    model: Any,
    angles: List[int] = [0, -90, 90, 180]
) -> List[Dict[str, Any]]:
    """
    Tests the drawing across multiple rotation angles to diagnose
    orientation-dependent ambiguity (e.g., 9 vs 6, or sideways 9 vs 5).

    Angles:
        0: As drawn
        -90: Rotated 90° Clockwise
        90: Rotated 90° Counter-Clockwise
        180: Upside down
    """
    # Extract grayscale PIL image first
    if raw_drawing.ndim == 3 and raw_drawing.shape[2] == 4:
        rgb = raw_drawing[:, :, :3]
        if np.any(rgb > 0):
            base_img = Image.fromarray(rgb).convert("L")
        else:
            base_img = Image.fromarray(raw_drawing[:, :, 3]).convert("L")
    elif raw_drawing.ndim == 3:
        base_img = Image.fromarray(raw_drawing).convert("L")
    else:
        base_img = Image.fromarray(raw_drawing.astype(np.uint8)).convert("L")

    angle_labels = {
        0: "As Drawn (0°)",
        -90: "Rotated 90° Clockwise",
        90: "Rotated 90° Counter-Clockwise",
        180: "Inverted (180°)"
    }

    results = []
    for angle in angles:
        rotated_img = base_img.rotate(angle) if angle != 0 else base_img
        tensor, preview_28 = preprocess_digit_image(np.array(rotated_img))

        probs = model.predict(tensor, verbose=0)[0]
        top_digit = int(np.argmax(probs))
        confidence = float(probs[top_digit]) * 100

        results.append({
            "angle": angle,
            "label": angle_labels.get(angle, f"{angle}°"),
            "predicted_digit": top_digit,
            "confidence": round(confidence, 2),
            "probabilities": probs,
            "preview_28x28": preview_28,
        })

    return results


def get_geometry_advisory(
    geometry: Dict[str, Any],
    angle_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Synthesizes stroke geometry and multi-angle results into a clear senior engineer advisory.
    """
    if geometry.get("is_empty"):
        return {"has_advisory": False, "message": "", "level": "info"}

    aspect = geometry.get("aspect_ratio", 1.0)
    as_drawn = angle_results[0] if angle_results else None

    # Check for sideways rotation:
    if aspect > 1.55 and as_drawn:
        # Find if a rotated version has very high confidence in a different digit
        cw_90 = next((r for r in angle_results if r["angle"] == -90), None)
        ccw_90 = next((r for r in angle_results if r["angle"] == 90), None)

        alt_suggestions = []
        if cw_90 and cw_90["confidence"] > 85.0 and cw_90["predicted_digit"] != as_drawn["predicted_digit"]:
            alt_suggestions.append(f"**Digit {cw_90['predicted_digit']}** ({cw_90['confidence']}% if rotated 90° Clockwise)")
        if ccw_90 and ccw_90["confidence"] > 85.0 and ccw_90["predicted_digit"] != as_drawn["predicted_digit"]:
            alt_suggestions.append(f"**Digit {ccw_90['predicted_digit']}** ({ccw_90['confidence']}% if rotated 90° Counter-Clockwise)")

        msg = (
            f"⚠️ **Horizontal Aspect Ratio Detected ({aspect}:1)**: "
            f"Standard upright digits are taller than wide. "
            f"At 0° orientation, this shape topologically resembles **Digit {as_drawn['predicted_digit']}**."
        )
        if alt_suggestions:
            msg += f"\n\n💡 **Orientation Insight:** When rotated upright, this matches: {' or '.join(alt_suggestions)}."

        return {
            "has_advisory": True,
            "message": msg,
            "level": "warning"
        }

    return {"has_advisory": False, "message": "", "level": "info"}
