import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    import cv2
    print("cv2 ok")
    import mediapipe as mp
    print("mediapipe ok")
    import torch
    print("torch ok")
    import albumentations
    print("albumentations ok")
    from src.signdetr_model import DETR
    print("DETR import ok")
    from src.utils.setup import get_classes
    print("get_classes ok")
    
    # Test model loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = DETR(num_classes=10) # Dummy num_classes
    print("DETR instantiation ok")
    
    # Test config loading
    classes = get_classes()
    print(f"Classes loaded: {len(classes)}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
