import os
import sys
import torch
import cv2
import numpy as np

def test_yolo_loading():
    """Test YOLO model loading with different methods"""
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    
    model_path = "yolov8n.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Will be downloaded when YOLO is imported.")
    
    # Method 1: Try with environment variable (simplest solution)
    print("\nMethod 1: Using environment variable")
    try:
        os.environ["TORCH_ALLOW_WEIGHTS_ONLY_SKIP"] = "1"
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✅ Success! Model loaded with environment variable method")
        return model
    except Exception as e:
        print(f"❌ Failed with error: {e}")
    
    # Method 2: Try with torch.load directly
    print("\nMethod 2: Using torch.load with weights_only=False")
    try:
        # This is the most direct way to bypass the security check
        # Only use with trusted models like the official YOLO models
        model_data = torch.load(model_path, weights_only=False)
        print("✅ Success! Model loaded with torch.load and weights_only=False")
        
        # Now try to create a YOLO model with this data
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print("✅ Success! YOLO model created")
            return model
        except Exception as e:
            print(f"❌ Failed to create YOLO model: {e}")
    except Exception as e:
        print(f"❌ Failed with error: {e}")
    
    # Method 3: Try with safe_globals
    print("\nMethod 3: Using safe_globals")
    try:
        from torch.serialization import safe_globals
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules import Conv, C2f, Detect
        from torch.nn.modules.container import Sequential
        
        with safe_globals([Sequential, DetectionModel, Conv, C2f, Detect]):
            from ultralytics import YOLO
            model = YOLO(model_path)
            print("✅ Success! Model loaded with safe_globals")
            return model
    except Exception as e:
        print(f"❌ Failed with error: {e}")
    
    # Method 4: Try with add_safe_globals
    print("\nMethod 4: Using add_safe_globals")
    try:
        from torch.serialization import add_safe_globals
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules import Conv, C2f, Detect
        from torch.nn.modules.container import Sequential
        
        add_safe_globals([Sequential, DetectionModel, Conv, C2f, Detect])
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✅ Success! Model loaded with add_safe_globals")
        return model
    except Exception as e:
        print(f"❌ Failed with error: {e}")
    
    # Method 5: Try with ultralytics.utils.TORCH_1_9 flag
    print("\nMethod 5: Using ultralytics.utils.TORCH_1_9 flag")
    try:
        import ultralytics.utils
        ultralytics.utils.TORCH_1_9 = True  # Force compatibility mode
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✅ Success! Model loaded with TORCH_1_9 flag")
        return model
    except Exception as e:
        print(f"❌ Failed with error: {e}")
    
    print("\n❌ All methods failed to load the YOLO model")
    return None

def test_detection():
    """Test object detection with the loaded model"""
    model = test_yolo_loading()
    
    if model is None:
        print("Cannot test detection because model loading failed")
        return
    
    # Try to run inference on a sample image
    try:
        # Create a simple test image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)  # Draw a green rectangle
        
        # Run inference
        results = model(img)
        print("\nInference test:")
        print(f"Detection results: {results}")
        print("✅ Inference successful!")
    except Exception as e:
        print(f"\n❌ Inference failed with error: {e}")

if __name__ == "__main__":
    test_detection()
