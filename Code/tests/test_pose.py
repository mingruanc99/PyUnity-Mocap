

import cv2
import numpy as np

def get_head_pose(landmarks_2d_points, img_w, img_h):
    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, 330.0, 65.0),          # Chin (Y Orig, Z Swapped)
        (225.0, -170.0, 135.0),      # Left eye left corner (X Swapped, Y Orig, Z Swapped)
        (-225.0, -170.0, 135.0),     # Right eye right corner (X Swapped, Y Orig, Z Swapped)
        (150.0, 150.0, 125.0),       # Left Mouth corner (X Swapped, Y Orig, Z Swapped)
        (-150.0, 150.0, 125.0)       # Right mouth corner (X Swapped, Y Orig, Z Swapped)
    ])

    image_points = np.array(landmarks_2d_points, dtype="double")

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
    
    return {
        "pitch": angles[0],
        "yaw": angles[1],
        "roll": angles[2]
    }

# Mock landmarks for a frontal face in a 640x480 image
# Nose: 1, Chin: 152, Left Eye Left: 33, Right Eye Right: 263, Left Mouth: 61, Right Mouth: 291
w, h = 640, 480
# Approximate positions for a face looking straight at the camera
# Nose at center
nose = (w/2, h/2) # 320, 240
# Chin (Ratio ~2: 80px down)
chin = (w/2, h/2 + 60) 
# Left Eye (40px up, 40px left)
# Wait, Model Left Eye (-225) is usually Image Right if mirrored?
# Standard: Left Eye (Image) corresponds to Right Eye (Person)?
# In MediaPipe: 
# 33 is Left Eye (Outer corner?). 263 is Right Eye.
# Let's assume non-mirrored for now.
# Left Eye (33) -> (w/2 - 40, h/2 - 40)
left_eye = (w/2 - 40, h/2 - 40)
# Right Eye (263) -> (w/2 + 40, h/2 - 40)
right_eye = (w/2 + 40, h/2 - 40)
# Mouth Left (61)
left_mouth = (w/2 - 20, h/2 + 30)
# Mouth Right (291)
right_mouth = (w/2 + 20, h/2 + 30)

landmarks = [nose, chin, left_eye, right_eye, left_mouth, right_mouth]

pose = get_head_pose(landmarks, w, h)
print(f"Pitch: {pose['pitch']}")
print(f"Yaw: {pose['yaw']}")
print(f"Roll: {pose['roll']}")
