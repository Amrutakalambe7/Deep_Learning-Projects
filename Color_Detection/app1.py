# Color Detection App (Detect colors from image, videos and live camera feed)
import streamlit as st
import cv2
import numpy as np
from utils import detect_colors
from PIL import Image
import tempfile

st.set_page_config(page_title="Color Detection App", layout="wide")

st.title("Color Detection App")
st.sidebar.title("Choose Input Source")
option = st.sidebar.radio("Select:", ['Camera', 'Upload Image', 'Upload Video'])

# ---------------------- CAMERA MODE ----------------------
if option == 'Camera':
    stframe = st.empty()

    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    start = st.button("Start Camera", key="start_camera")
    stop = st.button("Stop Camera", key="stop_camera_unique")

    if start:
        st.session_state.camera_active = True
    if stop:
        st.session_state.camera_active = False

    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        st.info("Camera is running... Click 'Stop Camera' to end.")

        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to open camera.")
                break

            frame = cv2.flip(frame, 1)
            result = detect_colors(frame)
            stframe.image(result, channels="BGR", use_container_width=True)

            # Use sleep to avoid overwhelming the CPU
            cv2.waitKey(1)

        cap.release()

# ---------------------- IMAGE MODE ----------------------
elif option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = detect_colors(image_np)
        st.image(result, caption="Detected Colors", channels="BGR")

# ---------------------- VIDEO MODE ----------------------
elif option == 'Upload Video':
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = detect_colors(frame)
            stframe.image(result, channels="BGR", use_container_width=True)

        cap.release()
