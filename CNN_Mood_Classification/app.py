import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import io

# Load trained model
model = load_model('mood_model.h5')

# Streamlit app UI
st.title("😊 Mood Classification App (Happy vs Sad)")
st.markdown("Upload an image, and the model will classify the mood.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((64, 64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    mood = "Happy 😊" if prediction > 0.5 else "Sad 😢"

    st.markdown(f"### Predicted Mood: {mood}")
