"""
evaluate.py
-----------
Comprehensive model evaluation and diagnostic suite ("The Analyser"):
1. Loads the trained CNN from models/best_digit_model.keras
2. Evaluates across 40,000 unseen EMNIST test images
3. Generates per-digit Precision, Recall, and F1-Scores (Classification Report)
4. Plots and exports a high-resolution Confusion Matrix heatmap
5. Extracts and plots an Error Gallery showing exact misclassified digits and model confidence
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf


def load_test_data():
    """Loads and prepares 40,000 EMNIST test images from local cache."""
    import emnist
    emnist.CACHE_FILE_PATH = str(ROOT_DIR / "data" / "emnist.zip")
    print("[1/4] Loading 40,000 EMNIST test images from cache...")
    x_test, y_test = emnist.extract_test_samples("digits")

    # Normalize to [0.0, 1.0] and add channel dimension
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    return x_test, y_test


def plot_confusion_matrix(cm, class_names, output_path):
    """Plots and saves a professional normalized confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    # Normalize by true label count (recall-oriented)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Prediction Rate"}
    )
    plt.title("EMNIST Digits - Normalized Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Digit", fontsize=12)
    plt.ylabel("Actual True Digit", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"      Saved Confusion Matrix plot: {output_path}")


def plot_misclassified_gallery(x_test, y_test, y_pred, y_probs, output_path, num_examples=12):
    """
    Identifies digits the model got wrong, sorted by the model's highest confidence
    in its wrong prediction (the most deceptive edge cases).
    """
    wrong_indices = np.where(y_pred != y_test)[0]
    if len(wrong_indices) == 0:
        print("      Zero misclassified samples found!")
        return

    # Sort wrong indices by the model's confidence in its wrong prediction (descending)
    wrong_confidences = [y_probs[i, y_pred[i]] for i in wrong_indices]
    sorted_order = np.argsort(wrong_confidences)[::-1]
    top_wrong_indices = wrong_indices[sorted_order[:num_examples]]

    cols = 4
    rows = int(np.ceil(num_examples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = axes.flatten()

    for idx, sample_i in enumerate(top_wrong_indices):
        img = x_test[sample_i].squeeze()
        true_label = y_test[sample_i]
        pred_label = y_pred[sample_i]
        confidence = y_probs[sample_i, pred_label] * 100

        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(
            f"True: {true_label} | Pred: {pred_label}\nConf: {confidence:.1f}%",
            color="red",
            fontsize=10
        )
        axes[idx].axis("off")

    # Hide any unused subplot axes
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Top Misclassified Test Digits (Model Blindspots)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      Saved Misclassified Gallery: {output_path}")


def evaluate_model():
    """Runs full evaluation and exports visual reports to reports/."""
    model_path = ROOT_DIR / "models" / "best_digit_model.keras"
    if not model_path.exists():
        print(f"[ERROR] Trained model not found at {model_path}")
        print("Please run 'python src/train.py' first.")
        return

    reports_dir = ROOT_DIR / "reports"
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load Data & Model
    x_test, y_test = load_test_data()

    print(f"[2/4] Loading model weights from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # 2. Run Inference
    print("[3/4] Running batch inference across 40,000 test images...")
    y_probs = model.predict(x_test, batch_size=256, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)

    # 3. Compute Metrics
    overall_accuracy = np.mean(y_pred == y_test)
    cm = confusion_matrix(y_test, y_pred)
    class_names = [str(i) for i in range(10)]
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)

    print("\n" + "=" * 60)
    print(f"  EMNIST DIGITS - COMPREHENSIVE EVALUATION REPORT")
    print(f"  Overall Test Accuracy: {overall_accuracy * 100:.2f}%")
    print("=" * 60)
    print(report)
    print("=" * 60)

    # 4. Save Artifacts
    print("\n[4/4] Generating diagnostic visualizations...")
    plot_confusion_matrix(cm, class_names, reports_dir / "confusion_matrix.png")
    plot_misclassified_gallery(x_test, y_test, y_pred, y_probs, reports_dir / "misclassified_gallery.png")

    # Save text report to file
    with open(reports_dir / "classification_report.txt", "w") as f:
        f.write(f"EMNIST Digits Test Accuracy: {overall_accuracy * 100:.2f}%\n\n")
        f.write(report)

    print("\nEvaluation complete! Check the 'reports/' directory for full visual diagnostics.")


if __name__ == "__main__":
    evaluate_model()
