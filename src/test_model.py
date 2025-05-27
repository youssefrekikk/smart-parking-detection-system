import os
import torch
import sys
from ultralytics.nn.tasks import DetectionModel

# Add DetectionModel to safe globals
torch.serialization.add_safe_globals([DetectionModel])

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
from parking_spot_annotation_model import ParkingSpotAnnotator

# Test the model
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_model.py <image_path> <model_path> [output_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Create model and load weights
    model = ParkingSpotAnnotator(model_path)
    
    # Detect parking spots with extremely low confidence threshold
    spots = model.detect_parking_spots(image_path=image_path, conf=0.4)
    
    # Visualize detections
    vis_image = model.visualize_detections(image_path=image_path, spots=spots)
    
    if output_path:
        import cv2
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to {output_path}")
        
        # Also save spots as JSON
        import json
        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump(spots, f, indent=2)
        print(f"Spots saved to {json_path}")
    else:
        import cv2
        cv2.imshow("Parking Spot Detection", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
