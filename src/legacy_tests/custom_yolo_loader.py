import os
import torch
import numpy as np
import cv2
from pathlib import Path
import urllib.request
import subprocess

class CustomYOLOLoader:
    """
    A custom loader for YOLO models that bypasses PyTorch 2.6+ security restrictions
    """
    def __init__(self, model_path="yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model using a direct approach"""
        try:
            # Try to use OpenCV's DNN module which doesn't have the PyTorch restrictions
            print(f"Loading YOLO model from {self.model_path} using OpenCV DNN")
            
            # Check if the model file exists
            if not os.path.exists(self.model_path):
                # Download the model if it doesn't exist
                self._download_model()
            
            # Convert to ONNX if needed
            onnx_path = self._ensure_onnx_model()
            
            # Load the model with OpenCV
            self.model = cv2.dnn.readNetFromONNX(onnx_path)
            
            # Set backend and target
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            print(f"Successfully loaded YOLO model using OpenCV DNN")
            return True
        except Exception as e:
            print(f"Error loading model with OpenCV DNN: {e}")
            return False
    
    def _download_model(self):
        """Download the YOLOv8 model"""
        print(f"Downloading YOLOv8 model to {self.model_path}")
        model_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{Path(self.model_path).name}"
        
        try:
            urllib.request.urlretrieve(model_url, self.model_path)
            print(f"Downloaded model from {model_url}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    
    def _ensure_onnx_model(self):
        """Ensure we have an ONNX version of the model"""
        onnx_path = str(Path(self.model_path).with_suffix('.onnx'))
        
        # Check if ONNX model already exists
        if os.path.exists(onnx_path):
            print(f"ONNX model already exists at {onnx_path}")
            return onnx_path
        
        # Convert PyTorch model to ONNX using a subprocess to avoid PyTorch loading issues
        print(f"Converting {self.model_path} to ONNX format")
        try:
            # Try using ultralytics export
            cmd = [
                "python", "-c", 
                f"from ultralytics import YOLO; YOLO('{self.model_path}').export(format='onnx')"
            ]
            
            # Set environment variable to allow loading
            env = os.environ.copy()
            env["TORCH_ALLOW_WEIGHTS_ONLY_SKIP"] = "1"
            
            # Run the conversion
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error during conversion: {result.stderr}")
                raise Exception(f"Failed to convert model: {result.stderr}")
            
            print(f"Successfully converted model to ONNX: {onnx_path}")
            return onnx_path
        except Exception as e:
            print(f"Error converting to ONNX: {e}")
            
            # If conversion fails, try to download pre-converted ONNX model
            try:
                print("Attempting to download pre-converted ONNX model")
                onnx_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{Path(onnx_path).name}"
                urllib.request.urlretrieve(onnx_url, onnx_path)
                print(f"Downloaded ONNX model from {onnx_url}")
                return onnx_path
            except Exception as e2:
                print(f"Error downloading ONNX model: {e2}")
                raise Exception(f"Failed to get ONNX model: {e2}")
    
    def detect(self, frame):
        """
        Run object detection on a frame
        Returns: list of detections with [x1, y1, x2, y2, confidence, class_id]
        """
        if self.model is None:
            print("Model not loaded")
            return []
        
        # Prepare image
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width = input_img.shape[:2]
        
        # Create blob from image (resize to 640x640)
        blob = cv2.dnn.blobFromImage(input_img, 1/255.0, (640, 640), swapRB=True, crop=False)
        
        # Set input to the model
        self.model.setInput(blob)
        
        # Forward pass
        outputs = self.model.forward()
        
        # Process outputs
        detections = []
        
        # YOLOv8 ONNX output format is different from PyTorch
        # It has shape (1, 84, num_boxes) where 84 = 4 (bbox) + 80 (class scores)
        for i in range(outputs.shape[1]):
            # Get detection data
            x1, y1, x2, y2 = outputs[0, i, 0:4]
            confidence = outputs[0, i, 4]
            
            if confidence < 0.25:  # Confidence threshold
                continue
            
            # Get class scores
            class_scores = outputs[0, i, 5:]
            class_id = np.argmax(class_scores)
            
            # Convert normalized coordinates to pixel values
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            
            detections.append([x1, y1, x2, y2, float(confidence), int(class_id)])
        
        return detections

    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        result_frame = frame.copy()
        
        for x1, y1, x2, y2, conf, class_id in detections:
            # Get class name
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(result_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_frame
