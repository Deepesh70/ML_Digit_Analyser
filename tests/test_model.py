import os
import numpy as np


def test_model_artifact_exists():
    """Check that the saved model file exists in models directory."""
    model_path = os.path.join("models", "best_digit_model.keras")
    assert os.path.exists(model_path), f"Model not found at {model_path}"

def test_model_prediction_shape():
    """Ensure the trained model loads and outputs 10 probability classes."""
    import tensorflow as tf
    model_path = os.path.join("models", "best_digit_model.keras")
    model = tf.keras.models.load_model(model_path)
    
    dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
    preds = model.predict(dummy_input)
    
    assert preds.shape == (1, 10)
    # Softmax output should sum to approximately 1.0
    assert np.isclose(np.sum(preds), 1.0, atol=1e-3)
