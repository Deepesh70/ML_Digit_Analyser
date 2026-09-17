import numpy as np
from src.analyser import analyze_stroke_geometry


def test_analyze_stroke_geometry_blank():
    """Ensure blank/empty drawing is properly detected."""
    blank = np.zeros((280, 280, 4), dtype=np.uint8)
    geometry = analyze_stroke_geometry(blank)
    assert geometry["is_empty"] is True
    assert geometry["width"] == 0
    assert geometry["height"] == 0


def test_analyze_stroke_geometry_drawn():
    """Ensure drawing dimensions and aspect ratio are calculated."""
    drawing = np.zeros((280, 280, 4), dtype=np.uint8)
    drawing[50:150, 100:150, :3] = 255  # Height 100, Width 50
    geometry = analyze_stroke_geometry(drawing)
    assert geometry["is_empty"] is False
    assert geometry["height"] > 0
    assert geometry["width"] > 0
    assert geometry["aspect_ratio"] > 0
