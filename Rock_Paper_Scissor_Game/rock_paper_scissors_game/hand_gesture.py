#Detect Hand_Gesture Module
def get_hand_gesture(landmarks):
    if not landmarks:
        return None
    # Thumb (landmark 4), Index (8), Middle (12)
    thumb = landmarks[4][0]
    index = landmarks[8][1]
    middle = landmarks[12][1]
    
    if index < landmarks[6][1] and middle < landmarks[10][1]:
        return "rock"
    elif index < landmarks[6][1] and middle > landmarks[10][1]:
        return "scissors"
    elif index > landmarks[6][1] and middle > landmarks[10][1]:
        return "paper"
    return "rock"
