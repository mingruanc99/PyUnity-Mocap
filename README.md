Video demo: https://youtu.be/zkUvEDPHdgE
# Key Features
## 1. Real-time Motion Capture (No Wearables)
Body Tracking: Utilizes MediaPipe Pose to track 33 body landmarks.
Inverse Kinematics (IK): Implements Unity Animation Rigging to map 2D webcam coordinates to 3D bone rotations, ensuring natural limb movements without expensive mocap suits.
Smooth Interpolation: Applies Mathf.Lerp and sliding window filters to reduce jitter and latency.

## 2. Facial Expression Driving
Face Mesh: Tracks 468 facial landmarks to calculate head pose (Pitch, Yaw, Roll) and facial blendshapes.
Emotion Sync: Real-time synchronization of blinking, mouth movements, and micro-expressions using aspect-ratio calculations.

## 3. ASL Recognition & Interaction
Static Gesture Recognition: Detects ASL alphabet (A-Z) and functional gestures (e.g., "Wave", "Point") using skeletal feature extraction and Machine Learning classifiers.
Interactive Feedback: Triggers voice responses (TTS) and specific animations (e.g., Waving, Bowing) in Unity upon detecting specific gestures.

# Tech Stack
Vision Computing (Server)
Language: Python 3.10
Core Libraries: OpenCV, MediaPipe, NumPy
ML/AI: Scikit-learn (for ASL Classifier), PyTorch (optional for deep learning extensions)

# Communication: UDP Socket (JSON payload)
3D Rendering (Client)
Engine: Unity 3D (URP/HDRP)
Animation: Unity Animator Controller, Animation Rigging Package (Inverse Kinematics)
Assets: VRM/FBX Models (Alhaitham Model), Mixamo Animations.

# Installation & Setup
## Prerequisites
- Python 3.10 installed.
- Unity Hub and Editor (Version 2021.3 LTS or later recommended).
- A standard Webcam.

Step 1: Python Environment (Vision Server)
1. Navigate to the code directory:
```
cd Code
```
2. Install dependencies:
```
pip install -r requirements.txt
```
(Note: Ensure mediapipe, opencv-python, and scikit-learn are included).

Step 2: Unity Project (Rendering Client)
1. Open Unity Hub.

2. Click Add and select the Unity folder.

3. Open the project. Note: It may take some time to import assets on the first launch.

4. Open the scene located at Assets/Scenes/DefaultScene.unity.

# Usage Guide
Start Unity: Press the ▶ Play button in the Unity Editor. The avatar should be in an Idle state, waiting for data.

Start Python Server: Run the main script in your terminal:
```
python main.py
```

# Calibration:
Sit directly in front of the webcam.
Ensure your face and upper body are visible.
The system will automatically detect landmarks and start driving the avatar.

# Interaction:

Move: Your head and hand movements will be mirrored.

ASL: Perform an ASL gesture (e.g., "Wave" hand) to trigger specific animations and audio.

# Configuration
UDP Settings:

Default IP: 127.0.0.1 (Localhost)

Default Port: 5052

To change these, modify main.py in Python and the UDPReceiver.cs component in Unity.

Sensitivity:

You can adjust Smoothing and Movement Multiplier in the Unity Inspector (AvatarController script) to fine-tune the avatar's responsiveness.

# Acknowledgments
Google MediaPipe for the robust perception pipeline.

HoYoverse for the character design inspiration (Alhaitham). Disclaimer: This project is for educational/research purposes only.

Mixamo for standard animation assets.


