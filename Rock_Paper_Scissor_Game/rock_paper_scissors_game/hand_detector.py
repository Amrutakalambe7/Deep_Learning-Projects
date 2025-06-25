#Hand Detector Module
import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, maxHands=1):
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=maxHands)
        self.draw = mp.solutions.drawing_utils

    def detect_hand(self, frame):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.draw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y) for lm in handLms.landmark]
                return landmarks
        return []
