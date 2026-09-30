import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


IMG = 128


st.set_page_config(
    page_title="Medical Image AI Prototype",
    page_icon="🩻"
)


st.title("🩻 Medical Image Classification Prototype")


st.warning(
    "Research/demo prototype only. "
    "This model uses synthetic training data by default, "
    "is not clinically validated, and must not be used "
    "for diagnosis or clinical decisions."
)


model = tf.keras.models.load_model(
    "artifacts/medical_classifier.keras"
)


st.info(
    "Demo note: The default model is trained on synthetic "
    "normal/abnormal image patterns. Results on unrelated "
    "real-world images are not meaningful."
)


uploaded_file = st.file_uploader(
    "Upload a PNG or JPG image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    resized = image.resize(
        (IMG, IMG)
    )

    x = np.array(
        resized
    ).astype("float32")

    probability = float(
        model.predict(
            x[None, ...],
            verbose=0
        )[0][0]
    )

    predicted_class = (
        "abnormal"
        if probability >= 0.5
        else "normal"
    )

    st.metric(
        "Abnormal-class probability",
        f"{probability:.2%}"
    )

    st.write(
        "Predicted class:",
        f"**{predicted_class}**"
    )

    st.caption(
        "This prediction is only a demonstration "
        "of the trained prototype and is not a medical diagnosis."
    )
