import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model

# =========================
# Path Setup
# =========================

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "cnn_model.h5"

# =========================
# Load Model
# =========================

model = load_model(MODEL_PATH)

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Digit Recognition")

st.title("🧠 Handwritten Digit Recognition")

uploaded = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:

    # Open with Pillow
    img = Image.open(uploaded).convert("L")

    st.subheader("Uploaded Image")
    st.image(img, width=200)

    # Preprocess
    img_resized = img.resize((28, 28))

    img_array = np.array(img_resized).astype("float32") / 255.0

    img_array = img_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(
        f"Predicted Digit: {predicted_digit}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    st.subheader("Prediction Probabilities")

    probabilities = prediction[0]

    for digit, prob in enumerate(probabilities):
        st.write(
            f"Digit {digit}: {prob * 100:.2f}%"
        )

    st.bar_chart(probabilities)
