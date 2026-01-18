import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions

import mediapipe as mp

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import socket
import json
import time
import math
from src.signdetr_model import DETR
from src.utils.boxes import rescale_bboxes
from src.utils.setup import get_classes
from src.asl_classifier import ASLClassifier
from src.threaded_inference import ThreadedInference
from src.two_handed_gestures import detect_two_handed_gestures, is_ext
from src.two_handed_gestures import detect_two_handed_gestures, is_ext
from src.two_handed_gestures import detect_two_handed_gestures

# --- CONFIGURATION ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "models/signdetr.pt"
OPENPOSE_PATH = "models/graph_opt.pb"
ASL_RF_PATH = "models/asl_model.p"
NEW_ASL_MODEL_PATH = "models/cnn8grps_rad1_model.h5"
WHITE_IMG_PATH = "models/white.jpg"
CONFIDENCE_THRESHOLD = 0.7 
OPENPOSE_THR = 0.2

# --- MEDIAPIPE SETUP ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Face Landmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)
face_landmarker = FaceLandmarker.create_from_options(options)

# Hand Landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)
hand_landmarker = HandLandmarker.create_from_options(options)

# --- SOCKET SETUP ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- SIGNDETR SETUP ---
classes = get_classes()
num_classes = len(classes)
detr_model = DETR(num_classes=num_classes)
# Check if model exists
if os.path.exists(MODEL_PATH):
    detr_model.load_pretrained(MODEL_PATH)
    print(f"SignDETR model loaded from {MODEL_PATH}")
else:
    print(f"WARNING: SignDETR Model not found at {MODEL_PATH}")
detr_model.to(DEVICE)
detr_model.eval()

# --- ASL CLASSIFIER SETUP ---
# Initialize the Unified ASL Classifier with both CNN (Primary) and RF (Legacy) models
asl_classifier = ASLClassifier(
    model_path_cnn=NEW_ASL_MODEL_PATH, 
    white_img_path=WHITE_IMG_PATH
)

# --- OPENPOSE SETUP ---
pose_net = None
if os.path.exists(OPENPOSE_PATH):
    pose_net = cv2.dnn.readNetFromTensorflow(OPENPOSE_PATH)
    print(f"OpenPose model loaded from {OPENPOSE_PATH}")
else:
    print(f"WARNING: OpenPose model not found at {OPENPOSE_PATH}")

# Transform for SignDETR
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# --- UTILS ---
def get_head_pose(landmarks, img_w, img_h):
    # 3D model points (Adjusted for Y-Down Coordinate System compatibility)
    # Nose (0,0,0)
    # Chin (Positive Y = Down)
    # Eyes (Negative Y = Up)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, 330.0, 65.0),          # Chin
        (-225.0, -170.0, 135.0),     # Left eye left corner
        (225.0, -170.0, 135.0),      # Right eye right corner
        (-150.0, 150.0, 125.0),      # Left Mouth corner
        (150.0, 150.0, 125.0)        # Right mouth corner
    ])

    # 2D image points from landmarks
    # Nose: 1, Chin: 152, Left Eye Left: 33, Right Eye Right: 263, Left Mouth: 61, Right Mouth: 291
    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),
        (landmarks[152].x * img_w, landmarks[152].y * img_h),
        (landmarks[33].x * img_w, landmarks[33].y * img_h),
        (landmarks[263].x * img_w, landmarks[263].y * img_h),
        (landmarks[61].x * img_w, landmarks[61].y * img_h),
        (landmarks[291].x * img_w, landmarks[291].y * img_h)
    ], dtype="double")

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    # Convert to Euler angles
    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    # Standard Mapping (OpenCV -> Unity)
    # OpenCV: Y-Down, Z-Forward
    # Unity: Y-Up, Z-Forward (Left Handed)
    return {
        "pitch": -angles[0],  # Pitch (X)
        "yaw": -angles[1],    # Yaw (Y)
        "roll": angles[2]     # Roll (Z)
    }

def detect_expression_blendshapes(blendshapes):
    bs_map = {b.category_name: b.score for b in blendshapes}
    
    # --- Debug Print ---
    debug_scores = {
        "smile": (bs_map.get('mouthSmileLeft', 0) + bs_map.get('mouthSmileRight', 0)) / 2.0,
        "frown": (bs_map.get('mouthFrownLeft', 0) + bs_map.get('mouthFrownRight', 0)) / 2.0,
        "blink": (bs_map.get('eyeBlinkLeft', 0) + bs_map.get('eyeBlinkRight', 0)) / 2.0,
        "jawOpen": bs_map.get('jawOpen', 0),
        "tongue": bs_map.get('tongueOut', 0),
        "brow_down": (bs_map.get('browDownLeft', 0) + bs_map.get('browDownRight', 0)) / 2.0,
        "brow_up": bs_map.get('browInnerUp', 0)
    }
    print(", ".join([f"{k}:{v:.2f}" for k, v in debug_scores.items()]))
    # --- End Debug Print ---
    
    smile_left = bs_map.get('mouthSmileLeft', 0)
    smile_right = bs_map.get('mouthSmileRight', 0)
    smile = (smile_left + smile_right) / 2.0
    
    frown_left = bs_map.get('mouthFrownLeft', 0)
    frown_right = bs_map.get('mouthFrownRight', 0)
    frown = (frown_left + frown_right) / 2.0
    
    brow_inner_up = bs_map.get('browInnerUp', 0)
    jaw_open = bs_map.get('jawOpen', 0)
    
    # New Blendshapes
    blink_l = bs_map.get('eyeBlinkLeft', 0)
    blink_r = bs_map.get('eyeBlinkRight', 0)
    tongue = bs_map.get('tongueOut', 0)
    brow_down_l = bs_map.get('browDownLeft', 0)
    brow_down_r = bs_map.get('browDownRight', 0)
    mouth_pucker = bs_map.get('mouthPucker', 0)

    expression = "neutral"
    
    if blink_l > 0.5 and blink_r > 0.5:
        expression = "blink"
    elif tongue > 0.5 and jaw_open > 0.3:
        expression = "tongue"
    elif mouth_pucker > 0.5:
        expression = "cute"
    elif brow_down_l > 0.3 and brow_down_r > 0.3:
        expression = "angry"
    elif brow_inner_up > 0.4 or (jaw_open > 0.3 and smile < 0.3):
        expression = "surprised"
    elif smile > 0.35:
        expression = "happy"
    elif frown > 0.3:
        expression = "sad"
        
    return expression, bs_map

def get_body_pose(net, image):
    if net is None:
        return []
        
    h, w = image.shape[:2]
    inWidth = 368
    inHeight = 368
    
    inp = cv2.dnn.blobFromImage(image, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[:, :19, :, :] 

    parts = []
    for i in range(18):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (w * point[0]) / out.shape[3]
        y = (h * point[1]) / out.shape[2]
        
        if conf > OPENPOSE_THR:
            parts.append({"id": i, "x": int(x), "y": int(y), "conf": float(conf)})
        else:
            parts.append(None)
            
    return parts

def detect_hand_shape(landmarks):
    """
    Detects static hand shapes like punch and gun based on distances from wrist.
    """
    wrist = landmarks[0]
    
    index_ext = is_ext(landmarks, 8, 6)
    middle_ext = is_ext(landmarks, 12, 10)
    ring_ext = is_ext(landmarks, 16, 14)
    pinky_ext = is_ext(landmarks, 20, 18)
    thumb_ext = is_ext(landmarks, 4, 2)
    
    # Hello: All fingers extended
    if index_ext and middle_ext and ring_ext and pinky_ext and thumb_ext:
        return "hello"

    # I Love You: Thumb, Index, and Pinky extended, others folded
    if thumb_ext and index_ext and pinky_ext and not middle_ext and not ring_ext:
        return "i_love_you"

    # Gun: Index and Thumb extended, others folded
    if index_ext and thumb_ext and middle_ext and not ring_ext and not pinky_ext:
        return "gun"
    
    # Punch: All fingers folded
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "punch"
        
    return "none"
# Gesture History
wrist_history = []
GESTURE_BUFFER_SIZE = 15

def detect_gesture(landmarks):
    global wrist_history
    wrist = landmarks[0]
    wrist_history.append((wrist.x, wrist.y))
    if len(wrist_history) > GESTURE_BUFFER_SIZE:
        wrist_history.pop(0)
    
    # 1. Check Hand Shape (Static)
    shape = detect_hand_shape(landmarks)
    if shape != "none":
        return shape

    # 2. Check Movement Gestures (Dynamic)
    if len(wrist_history) < GESTURE_BUFFER_SIZE:
        return "none"

    xs = [p[0] for p in wrist_history]
    ys = [p[1] for p in wrist_history]
    
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    
    if x_range > 0.2 and y_range < 0.1:
        return "wave_horizontal"
    if y_range > 0.2 and x_range < 0.1:
        return "wave_vertical"
        
    return "none"

def draw_text(img, text, pos, color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# --- MAIN LOOP ---
def main():
    cap = cv2.VideoCapture(0)
    
    print("Unified CV Server Started. Streaming to 127.0.0.1:5005...")
    
    threaded_inference = ThreadedInference(face_landmarker, hand_landmarker)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        threaded_inference.start(image_rgb)
        time.sleep(0.01) # Small delay to allow thread to start and hopefully finish
        face_results, hand_results = threaded_inference.get_results()
        
        if face_results is None or hand_results is None:
            continue
        
        data_packet = {
            "face_found": False,
            "head_pose": {"pitch":0, "yaw":0, "roll":0},
            "expression": "neutral",
            "hand_found": False,
            "gesture": "none",
            "sign_asl": "none",
            "sign_conf": 0.0,
            "asl_char": "none",
            "current_text": asl_classifier.current_str,
            "body_pose": []
        }

        # --- FACE LOGIC ---
        if face_results.face_landmarks:
            data_packet["face_found"] = True
            landmarks = face_results.face_landmarks[0]
            
            # Head Pose
            pose = get_head_pose(landmarks, w, h)
            data_packet["head_pose"] = pose
            
            # Expression
            if face_results.face_blendshapes:
                expr, _ = detect_expression_blendshapes(face_results.face_blendshapes[0])
                data_packet["expression"] = expr
            
            # Draw Face UI
            cv2.rectangle(image, (10, 10), (250, 120), (0, 0, 0), -1) # Background
            draw_text(image, f"Expr: {data_packet['expression']}", (20, 40), (0, 255, 255))
            draw_text(image, f"Pitch: {pose['pitch']:.1f}", (20, 70))
            draw_text(image, f"Yaw: {pose['yaw']:.1f}", (20, 95))
            draw_text(image, f"Roll: {pose['roll']:.1f}", (20, 120))


        # --- HAND LOGIC ---
        if hand_results.hand_landmarks:
            data_packet["hand_found"] = True
            
            two_handed_gesture = detect_two_handed_gestures(hand_results.hand_landmarks)
            if two_handed_gesture:
                data_packet["gesture"] = two_handed_gesture
            else:
                # Process single hand gestures
                hand_landmarks = hand_results.hand_landmarks[0]
                gesture = detect_gesture(hand_landmarks)
                data_packet["gesture"] = gesture
            
            # ASL CNN (New)
            asl_char = asl_classifier.predict_cnn(hand_results.hand_landmarks[0], w, h)
            data_packet["asl_char"] = asl_char
            
            # Draw Hand UI
            cv2.rectangle(image, (10, 150), (250, 240), (0, 0, 0), -1)
            draw_text(image, f"Gesture: {data_packet['gesture']}", (20, 170), (0, 255, 0))
            draw_text(image, f"ASL Char: {asl_char}", (20, 200), (255, 255, 0))
            draw_text(image, f"Text: {asl_classifier.current_str[-15:]}", (20, 230), (255, 255, 255))


        # --- BODY POSE LOGIC ---
        if pose_net:
            body_parts = get_body_pose(pose_net, image)
            # Serialize for JSON
            json_parts = []
            for bp in body_parts:
                if bp:
                    json_parts.append(bp)
                    cv2.circle(image, (bp['x'], bp['y']), 5, (0, 200, 200), -1)
                else:
                    json_parts.append(None)
            
            data_packet["body_pose"] = json_parts
            
            POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13]]
            for pair in POSE_PAIRS:
                partA = body_parts[pair[0]]
                partB = body_parts[pair[1]]
                if partA and partB:
                    cv2.line(image, (partA['x'], partA['y']), (partB['x'], partB['y']), (0, 200, 200), 2)


        # --- ASL LOGIC (SignDETR) ---
        transformed = transform(image=image_rgb)
        img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = detr_model(img_tensor)
        
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > CONFIDENCE_THRESHOLD
        
        if keep.any():
            max_idx = probas.max(-1).values.argmax()
            class_id = probas[max_idx].argmax()
            conf = probas[max_idx][class_id].item()
            sign_label = classes[class_id]
            
            data_packet["sign_asl"] = sign_label
            data_packet["sign_conf"] = conf

            kept_indices = keep.nonzero(as_tuple=True)[0]
            
            for idx in kept_indices:
                box_probas = probas[idx]
                box_class_id = box_probas.argmax()
                box_conf = box_probas[box_class_id].item()
                box_label = classes[box_class_id]
                
                raw_box = outputs['pred_boxes'][0, idx]
                cx, cy, bw, bh = raw_box.tolist()
                
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_text(image, f"{box_label} ({box_conf:.2f})", (x1, y1 - 10), (0, 255, 0))


        # Send Data
        print(f"Gesture: {data_packet['gesture']}, ASL Char: {data_packet['asl_char']}, Expression: {data_packet['expression']}")

        # Send Data
        try:
            json_str = json.dumps(data_packet)
            sock.sendto(json_str.encode(), (UDP_IP, UDP_PORT))
        except Exception as e:
            print(f"Socket Error: {e}")

        # Display
        cv2.imshow('Unified CV Server', image)
        key = cv2.waitKey(5) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('s'): # Speak
            print("Speak key pressed!")
            asl_classifier.speak_text()
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()