"""
app.py
------
Interactive Streamlit Web Application for the ML Digit Analyser.
Features:
- Live drawing canvas with real-time digit recognition
- Preprocessor transparency view (shows how the drawing was centered and scaled)
- Confidence probability distribution for all 10 digits
- File upload support for photos/scans of handwritten digits
- Model diagnostics & confusion matrix explorer
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf
from src.preprocess import preprocess_digit_image
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(
    page_title="ML Digit Analyser",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_trained_model():
    """Caches the trained model in memory for sub-second inference."""
    model_path = ROOT_DIR / "models" / "best_digit_model.keras"
    if not model_path.exists():
        return None
    return tf.keras.models.load_model(model_path)


def main():
    # Sidebar
    st.sidebar.title("🔢 Digit Analyser")
    st.sidebar.markdown(
        """
        **Model Details:**
        - **Dataset:** EMNIST Digits (240k train, 40k test)
        - **Architecture:** Deep CNN + BatchNorm + Dropout
        - **Test Accuracy:** `99.54%`
        ---
        """
    )
    st.sidebar.info(
        "💡 **Tip:** Draw in the center of the canvas with a solid stroke. "
        "The preprocessor automatically crops, scales, and centers the digit."
    )

    # Main App Header
    st.title("Handwritten Digit Analyser & Diagnostic Suite")
    st.markdown(
        "Draw a digit or upload an image to analyze the model's predictions and confidence calibration in real time."
    )

    # Load Model
    model = load_trained_model()
    if model is None:
        st.error(
            "⚠️ Model file not found at `models/best_digit_model.keras`.\n\n"
            "Please run `python src/train.py` in your terminal to train and export the model first."
        )
        return

    # Tabs
    tab_canvas, tab_upload, tab_analytics = st.tabs([
        "✍️ Interactive Canvas",
        "📁 Upload Image",
        "📊 Model Analytics & Diagnostics"
    ])

    # ==========================================
    # TAB 1: DRAWING CANVAS
    # ==========================================
    with tab_canvas:
        col_canvas, col_results = st.columns([1, 1], gap="large")

        with col_canvas:
            st.subheader("Draw a Digit (0 - 9)")
            stroke_width = st.slider("Brush Width:", min_value=10, max_value=30, value=18)

            # Interactive Canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                return_image_data=True,
                key="canvas",
            )

        with col_results:
            st.subheader("Analysis & Confidence")

            if canvas_result is not None and canvas_result.image_data is not None:
                # Check if canvas has any drawings
                raw_drawing = canvas_result.image_data
                is_empty = np.all(raw_drawing[:, :, :3] == 0)

                if is_empty:
                    st.info("Draw a number in the box to see real-time analysis.")
                else:
                    # Preprocess
                    tensor, preview_28x28 = preprocess_digit_image(raw_drawing)

                    # Predict
                    probabilities = model.predict(tensor, verbose=0)[0]
                    predicted_digit = int(np.argmax(probabilities))
                    confidence = float(probabilities[predicted_digit]) * 100

                    # Display Top Prediction
                    st.metric(
                        label="Predicted Digit",
                        value=f"{predicted_digit}",
                        delta=f"{confidence:.2f}% Confidence"
                    )

                    # Transparency Preview
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.caption("Centered 28×28 Tensor (Model Input)")
                        st.image(preview_28x28, width=120, clamp=True)
                    with col_p2:
                        top_3_indices = np.argsort(probabilities)[::-1][:3]
                        st.caption("Top 3 Predictions")
                        for rank, idx in enumerate(top_3_indices, 1):
                            st.write(f"**#{rank} Digit {idx}:** `{probabilities[idx] * 100:.2f}%`")

                    # Probability Bar Chart
                    st.markdown("#### Probability Distribution Across All Digits")
                    chart_df = pd.DataFrame({
                        "Digit": [str(i) for i in range(10)],
                        "Probability (%)": probabilities * 100
                    }).set_index("Digit")
                    st.bar_chart(chart_df, color="#3b82f6")

    # ==========================================
    # TAB 2: IMAGE UPLOAD
    # ==========================================
    with tab_upload:
        st.subheader("Upload a Photo or Scan of a Handwritten Digit")
        uploaded_file = st.file_uploader(
            "Choose an image file (PNG, JPG, JPEG):",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            col_u1, col_u2 = st.columns([1, 1], gap="large")
            with col_u1:
                img = Image.open(uploaded_file)
                st.image(img, caption="Original Uploaded Image", width=250)

                # Preprocess
                tensor, preview_28x28 = preprocess_digit_image(img)
                st.caption("Preprocessed 28×28 Model Input")
                st.image(preview_28x28, width=120, clamp=True)

            with col_u2:
                probabilities = model.predict(tensor, verbose=0)[0]
                predicted_digit = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_digit]) * 100

                st.metric(
                    label="Predicted Digit",
                    value=f"{predicted_digit}",
                    delta=f"{confidence:.2f}% Confidence"
                )

                chart_df = pd.DataFrame({
                    "Digit": [str(i) for i in range(10)],
                    "Probability (%)": probabilities * 100
                }).set_index("Digit")
                st.bar_chart(chart_df, color="#10b981")

    # ==========================================
    # TAB 3: MODEL ANALYTICS & DIAGNOSTICS
    # ==========================================
    with tab_analytics:
        st.subheader("EMNIST Test Set Diagnostic Suite")
        st.markdown(
            "Evaluation results calculated across **40,000 unseen test images** from the EMNIST Digits benchmark."
        )

        col_m1, col_m2 = st.columns(2)
        cm_path = ROOT_DIR / "reports" / "confusion_matrix.png"
        gallery_path = ROOT_DIR / "reports" / "misclassified_gallery.png"

        with col_m1:
            st.markdown("### Normalized Confusion Matrix")
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
            else:
                st.info("Run `python src/evaluate.py` to generate this matrix.")

        with col_m2:
            st.markdown("### Hardest Misclassified Examples")
            if gallery_path.exists():
                st.image(str(gallery_path), use_container_width=True)
            else:
                st.info("Run `python src/evaluate.py` to generate this gallery.")

        report_txt_path = ROOT_DIR / "reports" / "classification_report.txt"
        if report_txt_path.exists():
            st.markdown("### Full Classification Report (Precision, Recall, F1)")
            with open(report_txt_path, "r") as f:
                st.code(f.read(), language="text")


if __name__ == "__main__":
    main()
