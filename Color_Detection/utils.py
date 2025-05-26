import cv2
import numpy as np

# Define HSV ranges for basic colors (excluding white)
COLOR_RANGES = {
    'Red': [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],
    'Green': [(36, 50, 70), (89, 255, 255)],
    'Blue': [(90, 50, 70), (128, 255, 255)],
    'Yellow': [(20, 100, 100), (35, 255, 255)],
    'Orange': [(10, 100, 20), (20, 255, 255)],
    'Purple': [(129, 50, 70), (158, 255, 255)],
    'Pink': [(159, 50, 70), (169, 255, 255)],
    'Black': [(0, 0, 0), (180, 255, 30)],
    'Gray': [(0, 0, 40), (180, 50, 200)],
}

def detect_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    annotated = image.copy()

    for color, ranges in COLOR_RANGES.items():
        if color == 'Red':
            mask1 = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
            mask2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
            mask = mask1 | mask2
        else:
            mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # avoid noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(annotated, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated
