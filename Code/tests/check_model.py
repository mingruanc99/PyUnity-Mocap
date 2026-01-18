import torch
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))
from src.signdetr_model import DETR
from src.utils.setup import get_classes

try:
    classes = get_classes()
    num_classes = len(classes)
    print(f"Configured classes: {num_classes} ({classes})")
    
    model = DETR(num_classes=num_classes)
    print(f"Model initialized with num_classes={num_classes} (Output layer size: {num_classes + 1})")
    
    checkpoint_path = "models/signdetr.pt"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        print("Checkpoint loaded.")
        
        # Check specific layer match
        if 'linear_class.weight' in state_dict:
            ckpt_size = state_dict['linear_class.weight'].shape
            print(f"Checkpoint linear_class.weight shape: {ckpt_size}")
            
            model_size = model.linear_class.weight.shape
            print(f"Model linear_class.weight shape: {model_size}")
            
            if ckpt_size != model_size:
                print("MISMATCH DETECTED!")
        else:
            print("linear_class.weight not found in checkpoint keys: ", state_dict.keys())
            
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    else:
        print(f"Checkpoint file not found: {checkpoint_path}")

except Exception as e:
    print(f"Error: {e}")
