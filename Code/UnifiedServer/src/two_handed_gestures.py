import math

def get_dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_ext(landmarks, tip, pip):
    wrist = landmarks[0]
    return get_dist(landmarks[tip], wrist) > get_dist(landmarks[pip], wrist)

def detect_two_handed_gestures(hands_landmarks):
    if len(hands_landmarks) != 2:
        return None

    # Thank You: Both palms face up
    is_thank_you = True
    for hand_landmarks in hands_landmarks:
        # Check if hand is open
        if not (is_ext(hand_landmarks, 8, 6) and is_ext(hand_landmarks, 12, 10) and is_ext(hand_landmarks, 16, 14) and is_ext(hand_landmarks, 20, 18)):
            is_thank_you = False
            break
        
        # Check if palm is facing up
        wrist_y = hand_landmarks[0].y
        fingertip_y_avg = (hand_landmarks[8].y + hand_landmarks[12].y + hand_landmarks[16].y + hand_landmarks[20].y) / 4
        if fingertip_y_avg > wrist_y: # Y is inverted in image coordinates
            is_thank_you = False
            break
    
    if is_thank_you:
        return "thank_you"
        
    return None