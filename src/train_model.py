import os
import torch
import sys

# Set environment variable
os.environ["TORCH_ALLOW_WEIGHTS_ONLY_SKIP"] = "1"

# Patch torch.load to use weights_only=False
original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(f, *args, **kwargs)

# Replace torch.load with our patched version
torch.load = patched_torch_load

# Import the model after patching
from parking_spot_annotation_model import train_model_from_dataset

# Train the model
if __name__ == "__main__":
    epochs = 50
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except:
            pass
    
    # Use non-OBB model
    model = train_model_from_dataset("dataset", epochs=epochs, use_obb=False)
    
    if model:
        print(f"Model trained successfully!")
    else:
        print(f"Model training failed.")
