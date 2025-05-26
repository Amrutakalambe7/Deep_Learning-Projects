# Updated Streamlit App with Enhancements

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

st.set_page_config(page_title="Color Detection App", layout="wide")
st.title("🔍 Color Detection App")

st.sidebar.header("Detection Settings")
mode = st.sidebar.radio("Choose Detection Mode:", ["All Colors Except White", "Only Red", "Only Green", "Only Blue"])
use_camera = st.sidebar.toggle("Use Camera")

# Color thresholds in HSV
hsv_thresholds = {
    "red": [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],
    "green": [(36, 50, 70), (89, 255, 255)],
    "blue": [(90, 50, 70), (128, 255, 255)]
}

# Dynamic color sliders
st.sidebar.subheader("Adjust HSV Thresholds")
custom_h_min = st.sidebar.slider("Hue Min", 0, 180, 0)
custom_h_max = st.sidebar.slider("Hue Max", 0, 180, 180)
custom_s_min = st.sidebar.slider("Sat Min", 0, 255, 50)
custom_s_max = st.sidebar.slider("Sat Max", 0, 255, 255)
custom_v_min = st.sidebar.slider("Val Min", 0, 255, 50)
custom_v_max = st.sidebar.slider("Val Max", 0, 255, 255)

color_labels = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0)
}

def detect_colors(img, mode):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result_img = img.copy()
    start_time = time.time()

    if mode == "Only Red":
        masks = [cv2.inRange(hsv, hsv_thresholds['red'][0], hsv_thresholds['red'][1]),
                 cv2.inRange(hsv, hsv_thresholds['red'][2], hsv_thresholds['red'][3])]
        mask = cv2.bitwise_or(masks[0], masks[1])
        result_img = apply_mask_and_draw(img, mask, "red")

    elif mode == "Only Green":
        mask = cv2.inRange(hsv, hsv_thresholds['green'][0], hsv_thresholds['green'][1])
        result_img = apply_mask_and_draw(img, mask, "green")

    elif mode == "Only Blue":
        mask = cv2.inRange(hsv, hsv_thresholds['blue'][0], hsv_thresholds['blue'][1])
        result_img = apply_mask_and_draw(img, mask, "blue")

    elif mode == "All Colors Except White":
        for color in ["red", "green", "blue"]:
            if color == "red":
                masks = [cv2.inRange(hsv, hsv_thresholds['red'][0], hsv_thresholds['red'][1]),
                         cv2.inRange(hsv, hsv_thresholds['red'][2], hsv_thresholds['red'][3])]
                mask = cv2.bitwise_or(masks[0], masks[1])
            else:
                mask = cv2.inRange(hsv, hsv_thresholds[color][0], hsv_thresholds[color][1])
            result_img = apply_mask_and_draw(result_img, mask, color)

    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-5)
    cv2.putText(result_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    return result_img

def apply_mask_and_draw(img, mask, label):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), color_labels[label], 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_labels[label], 2)
    return img

def process_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = np.array(image.convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

col1, col2 = st.columns(2)

if use_camera:
    st.info("Press 'q' in the camera window to quit capturing.")
    if st.button("Start Camera"):
        cap = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            output_frame = detect_colors(frame, mode)
            FRAME_WINDOW.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
else:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = process_uploaded_image(uploaded_file)
        output = detect_colors(img, mode)
        col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
        col2.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Detected Colors", use_container_width=True)

        # Save output for download
        result_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        cv2.imwrite(result_path, output)
        with open(result_path, "rb") as file:
            btn = st.download_button(label="📥 Download Processed Image", data=file, file_name="processed_image.png", mime="image/png")
