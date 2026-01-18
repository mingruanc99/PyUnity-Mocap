# Unified CV Server

This project combines SignDETR (ASL Recognition) and Face/Hand tracking (MediaPipe) into a unified Python server that streams data to Unity via UDP.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Models:**
    Ensure the following models are present:
    - `models/signdetr.pt` (Pretrained SignDETR model)

3.  **Run Server:**
    ```bash
    python main.py
    ```
    The server will start the camera and stream JSON data to `127.0.0.1:5005`.

## Unity Integration

1.  Copy the `UnityScripts` folder into your Unity project's `Assets` folder.
2.  Attach `CVReceiver.cs` to an empty GameObject.
3.  Attach `AvatarController.cs` to your Avatar's root object.
4.  Link the references in `AvatarController` (Head Bone, Face Mesh, CVReceiver).

## Features

- **Head Pose:** Pitch, Yaw, Roll tracking.
- **Expressions:** Happy, Surprised, Wink (Heuristic based).
- **Gestures:** Wave Horizontal/Vertical.
- **ASL:** Recognizes signs (Hello, I Love You, Thank You) using SignDETR.
