import sys
import os
import traceback

sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    import mediapipe as mp
    print(f"MediaPipe version: {mp.__version__}")
    try:
        print("Checking mp.tasks...")
        t = mp.tasks
        print("mp.tasks ok")
        v = mp.tasks.vision
        print("mp.tasks.vision ok")
    except AttributeError:
        print("ERROR: mp.tasks or mp.tasks.vision missing. Upgrade mediapipe.")
        
    import albumentations as A
    print(f"Albumentations version: {A.__version__}")
    
    # Check Transform
    try:
        from albumentations.pytorch import ToTensorV2
        t = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])
        print("Albumentations Compose ok")
    except Exception as e:
        print(f"Albumentations Error: {e}")

except Exception as e:
    print(f"General Error: {e}")
    traceback.print_exc()
