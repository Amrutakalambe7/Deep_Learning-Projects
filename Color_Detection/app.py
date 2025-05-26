import streamlit as st
import cv2
import numpy as np
from utils import detect_colors
from PIL import Image

st.set_page_config(page_title="Color Detection App", layout="wide")

st.title("Color Detection App (Except White)")
st.sidebar.title("Choose Input Source")
option = st.sidebar.radio("Select:", ['Camera', 'Upload Image'])

if option == 'Camera':
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    st.info("Press 'Stop' in the top-right to release the camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to open camera.")
            break
        frame = cv2.flip(frame, 1)
        result = detect_colors(frame)
        stframe.image(result, channels="BGR")
        if st.button("Stop"):
            cap.release()
            break

elif option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = detect_colors(image_np)
        st.image(result, caption="Detected Colors", channels="BGR")
