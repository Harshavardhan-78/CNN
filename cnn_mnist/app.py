import streamlit as st
import numpy as np
import cv2

from tensorflow.keras.models import load_model

# Load model
model = load_model("models/cnn_model.h5")

st.set_page_config(page_title="Digit Recognition")

st.title("🧠 Handwritten Digit Recognition")

uploaded = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:

    # Read image
    file_bytes = np.asarray(
        bytearray(uploaded.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_GRAYSCALE
    )

    st.subheader("Uploaded Image")
    st.image(img, width=200)

    # Preprocess
    resized = cv2.resize(img, (28, 28))

    resized = resized.astype("float32") / 255.0

    resized = resized.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(resized, verbose=0)

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
            f"Digit {digit}: {prob*100:.2f}%"
        )

    st.bar_chart(probabilities)