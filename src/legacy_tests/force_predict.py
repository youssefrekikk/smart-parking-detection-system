import os
import torch
import sys
import cv2
import numpy as np
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

# Force predictions with extremely low confidence
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python force_predict.py <image_path> <model_path> [output_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Create model and load weights
    model = ParkingSpotAnnotator(model_path)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        sys.exit(1)
    
    # Create visualization
    vis_image = image.copy()
    
    # Try direct model inference with extremely low confidence
    try:
        results = model.model(image, conf=0.001, verbose=False)
        
        # Check if we got any results
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Draw all detections, even very low confidence ones
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    try:
                        # Get confidence
                        conf = float(box.conf[0])
                        
                        # Get box coordinates
                        xyxyxyxy = box.xyxyxyxy[0].cpu().numpy()
                        points = xyxyxyxy.reshape(4, 2).astype(np.int32)
                        
                        # Choose color based on confidence (red for low, green for high)
                        color = (0, int(255 * min(conf * 10, 1)), int(255 * (1 - min(conf * 10, 1))))
                        
                        # Draw the box
                        cv2.polylines(vis_image, [points], True, color, 2)
                        
                        # Add confidence text
                        cv2.putText(
                            vis_image,
                            f"{conf:.3f}",
                            (points[0][0], points[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )
                    except Exception as e:
                        print(f"Error processing box: {e}")
        else:
            print("No detections found with model, trying heuristic detection")
            
            # Use our heuristic detection
            spots = model._heuristic_detection(image)
            
            # Draw the heuristic detections
            if spots:
                boxes = spots.get("boxes", [])
                img_height, img_width = image.shape[:2]
                
                for i, box in enumerate(boxes):
                    # Get coordinates
                    x1 = box.get("x1", 0) * img_width
                    y1 = box.get("y1", 0) * img_height
                    x2 = box.get("x2", 0) * img_width
                    y2 = box.get("y2", 0) * img_height
                    angle = box.get("angle", 0)
                    
                    # Calculate center, width, height
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    
                    # Create rotated rectangle
                    rect = ((center_x, center_y), (width, height), angle)
                    box_points = cv2.boxPoints(rect)
                    box_points = np.intp(box_points)
                    
                    # Draw the box (use blue for heuristic detections)
                    cv2.drawContours(vis_image, [box_points], 0, (255, 0, 0), 2)
                    
                    # Add spot ID
                    cv2.putText(
                        vis_image,
                        f"H#{i+1}",
                        (int(center_x), int(center_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )
    except Exception as e:
        print(f"Error during model inference: {e}")
        import traceback
        traceback.print_exc()
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to {output_path}")
    else:
        cv2.imshow("All Detections", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
