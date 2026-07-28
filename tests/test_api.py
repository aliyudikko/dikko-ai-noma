# test_checkpoint_load.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.model.config import ModelConfig
from src.model.model import DikkoHausaLM
import pickle
import torch

def test_load():
    checkpoint_path = "checkpoints/checkpoint_finetune.pt"
    
    # Method 1: Try torch.load
    print("Method 1: torch.load with weights_only=False")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print("✅ Success!")
        print(f"Keys: {checkpoint.keys()}")
        if "model_config" in checkpoint:
            print(f"Config type: {type(checkpoint['model_config'])}")
            print(f"Config: {checkpoint['model_config'].__dict__}")
        return checkpoint
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Method 2: Custom unpickler
    print("\nMethod 2: Custom unpickler")
    try:
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'ModelConfig':
                    return ModelConfig
                if name == 'DikkoHausaLM':
                    return DikkoHausaLM
                return super().find_class(module, name)
        
        with open(checkpoint_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            checkpoint = unpickler.load()
        print("✅ Success!")
        return checkpoint
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

if __name__ == "__main__":
    test_load()