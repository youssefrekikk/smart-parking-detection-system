import os
os.environ["TORCH_ALLOW_WEIGHTS_ONLY_SKIP"] = "1"
import cv2
import numpy as np
import torch
import json
import logging
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import random
from transform_to_obb import transform_detections_to_obb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParkingSpotAnnotator:
    """Model to automatically detect and annotate parking spots"""
    
    def __init__(self, model_path=None):
        """Initialize the parking spot annotator
        
        Args:
            model_path: Path to a pre-trained YOLOv8 model, if available
        """
        self.model = None
        self.model_path = model_path
        
        # Try to load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
    
    def train(self, dataset_dir, epochs=50, img_size=640, batch_size=16, use_obb=False):
        """Train a YOLOv8 model on the parking spot dataset
        
        Args:
            dataset_dir: Directory containing the dataset (images and labels)
            epochs: Number of training epochs
            img_size: Image size for training
            batch_size: Batch size for training
            use_obb: Whether to use OBB (Oriented Bounding Box) model
        
        Returns:
            Path to the trained model
        """
        try:
            # Create a YAML file for the dataset
            dataset_yaml = self._create_dataset_yaml(dataset_dir)
            
            # Initialize a new YOLOv8 model (non-OBB)
            if use_obb:
                logger.info("Using YOLOv8m-obb.pt model for oriented bounding boxes")
                self.model = YOLO('yolov8m-obb.pt')
            else:
                logger.info("Using standard YOLOv8m.pt model")
                self.model = YOLO('yolov8m.pt')  # Use medium model for better performance
            
            # Train the model
            logger.info(f"Starting training for {epochs} epochs...")
            results = self.model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                patience=15,  # Early stopping patience
                device='0' if torch.cuda.is_available() else 'cpu'
            )
            
            # Save the trained model
            self.model_path = results.save_dir
            logger.info(f"Training completed. Model saved to {self.model_path}")
            
            return self.model_path
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _create_dataset_yaml(self, dataset_dir):
        """Create a YAML file for the dataset
        
        Args:
            dataset_dir: Directory containing the dataset
        
        Returns:
            Path to the created YAML file
        """
        # Load class names from notes.json
        notes_path = os.path.join(dataset_dir, 'notes.json')
        if os.path.exists(notes_path):
            with open(notes_path, 'r') as f:
                notes = json.load(f)
            
            # Extract class names
            names = [category['name'] for category in notes.get('categories', [])]
        else:
            # Default class name if notes.json is not available
            names = ['Parking Spot']
        
        # Create dataset YAML content
        yaml_content = {
            'path': os.path.abspath(dataset_dir),
            'train': 'images',  # Assuming all images are used for training
            'val': 'images',    # Using same images for validation (not ideal but works for small dataset)
            'names': names
        }
        
        # Write YAML file
        yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            f.write(f"path: {yaml_content['path']}\n")
            f.write(f"train: {yaml_content['train']}\n")
            f.write(f"val: {yaml_content['val']}\n")
            f.write("names:\n")
            for i, name in enumerate(yaml_content['names']):
                f.write(f"  {i}: {name}\n")
        
        logger.info(f"Created dataset YAML at {yaml_path}")
        return yaml_path

    def detect_parking_spots(self, image_path=None, image=None, conf=0.1):
        """Detect parking spots in an image and transform to OBB"""
        if self.model is None:
            logger.error("No model available for detection")
            return self._fallback_detection(image_path, image)
        
        try:
            # Run inference with regular YOLO
            if image_path:
                results = self.model(image_path, conf=conf)
                image = cv2.imread(image_path)
            elif image is not None:
                results = self.model(image, conf=conf)
            else:
                logger.error("Either image_path or image must be provided")
                return None
            
            # Get image dimensions
            img_height, img_width = image.shape[:2]
            
            # Process regular YOLO results
            regular_detections = []
            for i, result in enumerate(results):
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                for j, box in enumerate(boxes):
                    # Get regular bounding box
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Normalize coordinates
                    x1_norm = x1 / img_width
                    y1_norm = y1 / img_height
                    x2_norm = x2 / img_width
                    y2_norm = y2 / img_height
                    
                    detection = {
                        "id": j + 1,
                        "x1": float(x1_norm),
                        "y1": float(y1_norm),
                        "x2": float(x2_norm),
                        "y2": float(y2_norm),
                        "confidence": float(box.conf[0])
                    }
                    regular_detections.append(detection)
            
            # Transform regular detections to OBB
            if regular_detections:
                obb_detections = transform_detections_to_obb(regular_detections, image, method="contour_analysis")
                
                # Add group information
                parking_spots = []
                for detection in obb_detections:
                    center_y = (detection["y1"] + detection["y2"]) / 2
                    group_id = f"group_{int(center_y * 3) + 1}"
                    
                    spot = {
                        "id": detection["id"],
                        "x1": detection["x1"],
                        "y1": detection["y1"],
                        "x2": detection["x2"],
                        "y2": detection["y2"],
                        "angle": detection["angle"],
                        "group_id": group_id,
                        "corners": detection["corners"],
                        "confidence": detection["confidence"]
                    }
                    parking_spots.append(spot)
                
                return self._format_spots_for_frontend(parking_spots, img_width, img_height)
            
            # If no spots detected, use fallback
            else:
                logger.warning("No parking spots detected, using fallback method")
                return self._fallback_detection(image_path, image)
                
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._fallback_detection(image_path, image)

    def _heuristic_detection(self, image):
        """Heuristic-based parking spot detection
        
        This method uses computer vision techniques to detect potential parking spots
        """
        logger.info("Using heuristic-based detection")
        
        if image is None:
            logger.error("No valid image for heuristic detection")
            return self._create_grid_spots(640, 480)  # Default size
        
        img_height, img_width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 19, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        parking_spots = []
        spot_id = 1
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (adjust these thresholds based on your images)
            min_area = img_width * img_height * 0.005  # 0.5% of image
            max_area = img_width * img_height * 0.1    # 10% of image
            
            if min_area <= area <= max_area:
                # Get rotated rectangle
                rect = cv2.minAreaRect(contour)
                center, (width, height), angle = rect
                
                # Filter by aspect ratio
                aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                if aspect_ratio > 5:  # Skip very elongated shapes
                    continue
                
                # Normalize center, width, height
                center_norm = (center[0] / img_width, center[1] / img_height)
                width_norm = width / img_width
                height_norm = height / img_height
                
                # Calculate x1, y1, x2, y2 (normalized)
                x1 = max(0, center_norm[0] - width_norm / 2)
                y1 = max(0, center_norm[1] - height_norm / 2)
                x2 = min(1, center_norm[0] + width_norm / 2)
                y2 = min(1, center_norm[1] + height_norm / 2)
                
                # Get box points
                box_points = cv2.boxPoints(rect)
                
                # Normalize points
                box_points_norm = box_points.copy()
                box_points_norm[:, 0] /= img_width
                box_points_norm[:, 1] /= img_height
                
                # Assign to a group based on position
                group_id = f"group_{int(center_norm[1] * 3) + 1}"
                
                # Create spot dictionary
                spot = {
                    "id": spot_id,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "angle": float(angle),
                    "group_id": group_id,
                    "corners": box_points_norm.tolist(),
                    "confidence": 0.5  # Default confidence for heuristic
                }
                
                parking_spots.append(spot)
                spot_id += 1
        
        # If we found at least 3 spots, use them
        if len(parking_spots) >= 3:
            return self._format_spots_for_frontend(parking_spots, img_width, img_height)
        
        # Try another approach - line detection
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is not None and len(lines) > 10:
            # Group lines by angle and proximity
            parking_spots = []
            spot_id = 1
            
            # Create parking spots from parallel lines
            for i in range(0, min(len(lines), 30), 2):  # Process pairs of lines
                if i+1 >= len(lines):
                    break
                
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[i+1][0]
                
                # Calculate center point between the two lines
                center_x = (x1 + x2 + x3 + x4) / 4
                center_y = (y1 + y2 + y3 + y4) / 4
                
                # Estimate width and height
                width = max(abs(x2 - x1), abs(x4 - x3))
                height = max(abs(y2 - y1), abs(y4 - y3))
                
                # Ensure minimum size
                width = max(width, 30)
                height = max(height, 30)
                
                # Calculate angle
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # Normalize center, width, height
                center_norm = (center_x / img_width, center_y / img_height)
                width_norm = width / img_width
                height_norm = height / img_height
                
                # Calculate x1, y1, x2, y2 (normalized)
                x1 = max(0, center_norm[0] - width_norm / 2)
                y1 = max(0, center_norm[1] - height_norm / 2)
                x2 = min(1, center_norm[0] + width_norm / 2)
                y2 = min(1, center_norm[1] + height_norm / 2)
                
                # Create rotated rectangle
                rect = ((center_x, center_y), (width, height), angle)
                box_points = cv2.boxPoints(rect)
                
                # Normalize points
                box_points_norm = box_points.copy()
                box_points_norm[:, 0] /= img_width
                box_points_norm[:, 1] /= img_height
                
                # Assign to a group based on position
                group_id = f"group_{int(center_norm[1] * 3) + 1}"
                
                # Create spot dictionary
                spot = {
                    "id": spot_id,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "angle": float(angle),
                    "group_id": group_id,
                    "corners": box_points_norm.tolist(),
                    "confidence": 0.5  # Default confidence for heuristic
                }
                
                parking_spots.append(spot)
                spot_id += 1
            
            if len(parking_spots) >= 3:
                return self._format_spots_for_frontend(parking_spots, img_width, img_height)
        
        # If all else fails, fall back to grid
        logger.info("Heuristic detection failed, falling back to grid")
        return self._create_grid_spots(img_width, img_height)

    def _fallback_detection(self, image_path=None, image=None):
        """Fallback method for detecting parking spots when model fails
        
        This uses simple image processing techniques to detect potential parking spots
        """
        logger.info("Using fallback detection method")
        
        # Load image if path provided
        if image_path:
            image = cv2.imread(image_path)
        
        if image is None:
            logger.error("No valid image for fallback detection")
            return None
        
        img_height, img_width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_area = (img_width * img_height) * 0.005  # Min 0.5% of image area
        max_area = (img_width * img_height) * 0.05   # Max 5% of image area
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                valid_contours.append(contour)
        
        # If not enough contours found, create a grid of parking spots
        if len(valid_contours) < 5:
            logger.info("Not enough contours found, creating grid")
            return self._create_grid_spots(img_width, img_height)
        
        # Process contours to create parking spots
        parking_spots = []
        for i, contour in enumerate(valid_contours):
            # Get rotated rectangle
            rect = cv2.minAreaRect(contour)
            center, (width, height), angle = rect
            
            # Skip if aspect ratio is too extreme
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            if aspect_ratio > 4:
                continue
            
            # Normalize center, width, height
            center_norm = (center[0] / img_width, center[1] / img_height)
            width_norm = width / img_width
            height_norm = height / img_height
            
            # Calculate x1, y1, x2, y2 (normalized)
            x1 = max(0, center_norm[0] - width_norm / 2)
            y1 = max(0, center_norm[1] - height_norm / 2)
            x2 = min(1, center_norm[0] + width_norm / 2)
            y2 = min(1, center_norm[1] + height_norm / 2)
            
            # Get box points
            box_points = cv2.boxPoints(rect)
            
            # Normalize points
            box_points_norm = box_points.copy()
            box_points_norm[:, 0] /= img_width
            box_points_norm[:, 1] /= img_height
            
            # Assign to a group based on position
            group_id = f"group_{int(center_norm[1] * 3) + 1}"
            
            # Create spot dictionary
            spot = {
                "id": i + 1,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "angle": float(angle),
                "group_id": group_id,
                "corners": box_points_norm.tolist(),
                "confidence": 0.5  # Default confidence for fallback
            }
            
            parking_spots.append(spot)
        
        # If still not enough spots, create a grid
        if len(parking_spots) < 5:
            logger.info("Not enough valid spots found, creating grid")
            return self._create_grid_spots(img_width, img_height)
        
        return self._format_spots_for_frontend(parking_spots, img_width, img_height)

    def _create_grid_spots(self, img_width, img_height):
        """Create a grid of parking spots"""
        logger.info("Creating grid of parking spots")
        
        # Define grid parameters
        rows = 3
        cols = 4
        
        # Calculate spot dimensions
        spot_width = 1.0 / cols * 0.8  # 80% of cell width
        spot_height = 1.0 / rows * 0.8  # 80% of cell height
        
        parking_spots = []
        spot_id = 1
        
        for row in range(rows):
            group_id = f"group_{row + 1}"
            
            for col in range(cols):
                # Calculate center position (normalized)
                center_x = (col + 0.5) / cols
                center_y = (row + 0.5) / rows
                
                # Calculate corners (normalized)
                x1 = center_x - spot_width / 2
                y1 = center_y - spot_height / 2
                x2 = center_x + spot_width / 2
                y2 = center_y + spot_height / 2
                
                # Create corners for visualization
                corners = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ])
                
                # Create spot dictionary
                spot = {
                    "id": spot_id,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "angle": 0.0,
                    "group_id": group_id,
                    "corners": corners.tolist(),
                    "confidence": 0.5  # Default confidence for grid
                }
                
                parking_spots.append(spot)
                spot_id += 1
        
        return self._format_spots_for_frontend(parking_spots, img_width, img_height)

    def _format_spots_for_frontend(self, spots, img_width, img_height):
        """Format spots for the frontend

        Args:
            spots: List of detected parking spots
            img_width: Image width
            img_height: Image height
            
        Returns:
            Dictionary with boxes and groups in the format expected by the frontend
        """
        boxes = []
        groups = {}

        for spot in spots:
            # Create box entry
            box = {
                "x1": spot["x1"],
                "y1": spot["y1"],
                "x2": spot["x2"],
                "y2": spot["y2"],
                "angle": spot["angle"],
                "group_id": spot["group_id"]
            }
            boxes.append(box)
            
            # Track groups
            if spot["group_id"] not in groups:
                group_id = spot["group_id"]
                # Extract group number from group_id (e.g., "group_1" -> 1)
                try:
                    group_num = int(group_id.split('_')[1])
                    location = f"Section {group_num}"
                except:
                    location = "Unknown Section"
                
                groups[group_id] = {
                    "group_id": group_id,
                    "location": location,
                    "box_indices": []
                }
            
            # Add this box index to the group
            groups[spot["group_id"]]["box_indices"].append(len(boxes) - 1)

        # Convert group map to list
        groups_list = list(groups.values())

        return {
            "camera_id": "auto_generated",
            "boxes": boxes,
            "groups": groups_list
        }

    def learn_from_corrections(self, original_spots, corrected_spots, image_path=None, image=None):
        """Learn from admin corrections using reinforcement learning
        
        Args:
            original_spots: Original spots detected by the model
            corrected_spots: Corrected spots provided by the admin
            image_path: Path to the image (optional)
            image: Image as numpy array (optional)
            
        Returns:
            True if learning was successful, False otherwise
        """
        logger.info("Learning from admin corrections")
        
        try:
            # If we don't have a model yet, we can't learn
            if self.model is None:
                logger.warning("No model available for learning")
                return False
            
            # Extract the original and corrected boxes
            original_boxes = original_spots.get("boxes", [])
            corrected_boxes = corrected_spots.get("boxes", [])
            
            # If no corrections were made, nothing to learn
            if len(original_boxes) == len(corrected_boxes) and all(
                self._boxes_are_similar(orig, corr) 
                for orig, corr in zip(original_boxes, corrected_boxes)
            ):
                logger.info("No significant corrections detected")
                return True
            
            # Load the image if needed for fine-tuning
            if image is None and image_path:
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Failed to load image from {image_path}")
                    return False
            
            # Convert corrected boxes to YOLO format for fine-tuning
            if image is not None:
                img_height, img_width = image.shape[:2]
                
                # Create a temporary directory for fine-tuning
                import tempfile
                import shutil
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Create images and labels directories
                    images_dir = os.path.join(temp_dir, "images")
                    labels_dir = os.path.join(temp_dir, "labels")
                    os.makedirs(images_dir, exist_ok=True)
                    os.makedirs(labels_dir, exist_ok=True)
                    
                    # Save the image
                    image_filename = "correction.jpg"
                    image_path = os.path.join(images_dir, image_filename)
                    cv2.imwrite(image_path, image)
                    
                    # Create label file in YOLO format
                    label_filename = "correction.txt"
                    label_path = os.path.join(labels_dir, label_filename)
                    
                    with open(label_path, "w") as f:
                        for box in corrected_boxes:
                            # Get the corners (either from the box or calculate them)
                            if "corners" in box:
                                corners = box["corners"]
                            else:
                                # Calculate corners from x1, y1, x2, y2, angle
                                x1, y1 = box["x1"], box["y1"]
                                x2, y2 = box["x2"], box["y2"]
                                angle = box.get("angle", 0)
                                
                                # Convert to pixel coordinates
                                x1_px = x1 * img_width
                                y1_px = y1 * img_height
                                x2_px = x2 * img_width
                                y2_px = y2 * img_height
                                
                                # Calculate center, width, height
                                center_x = (x1_px + x2_px) / 2
                                center_y = (y1_px + y2_px) / 2
                                width = abs(x2_px - x1_px)
                                height = abs(y2_px - y1_px)
                                
                                # Create rotated rectangle
                                rect = ((center_x, center_y), (width, height), angle)
                                corners_px = cv2.boxPoints(rect)
                                
                                # Normalize corners
                                corners = corners_px.copy()
                                corners[:, 0] /= img_width
                                corners[:, 1] /= img_height
                            
                            # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                            # For parking spots, class_id is 0
                            line = "0 "
                            for corner in corners:
                                line += f"{corner[0]} {corner[1]} "
                            f.write(line.strip() + "\n")
                    
                    # Create dataset YAML
                    yaml_path = os.path.join(temp_dir, "dataset.yaml")
                    with open(yaml_path, "w") as f:
                        f.write(f"path: {temp_dir}\n")
                        f.write("train: images\n")
                        f.write("val: images\n")
                        f.write("names:\n")
                        f.write("  0: Parking Spot\n")
                    
                    # Fine-tune the model with the corrected data
                    logger.info("Fine-tuning model with corrected data")
                    
                    # Use a higher learning rate for fine-tuning
                    results = self.model.train(
                        data=yaml_path,
                        epochs=10,  # Fewer epochs for fine-tuning
                        imgsz=640,
                        batch=1,  # Small batch size for fine-tuning
                        patience=5,
                        lr0=0.001,  # Higher learning rate
                        device='0' if torch.cuda.is_available() else 'cpu'
                    )
                    
                    # Save the fine-tuned model
                    self.model_path = results.save_dir
                    logger.info(f"Fine-tuning completed. Model saved to {self.model_path}")
                
                return True
            
            else:
                logger.error("No image available for fine-tuning")
                return False
                
        except Exception as e:
            logger.error(f"Error during learning from corrections: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _boxes_are_similar(self, box1, box2, threshold=0.05):
        """Check if two boxes are similar
        
        Args:
            box1: First box
            box2: Second box
            threshold: Threshold for considering boxes similar
            
        Returns:
            True if boxes are similar, False otherwise
        """
        # Check if coordinates are within threshold
        if (abs(box1.get("x1", 0) - box2.get("x1", 0)) > threshold or
            abs(box1.get("y1", 0) - box2.get("y1", 0)) > threshold or
            abs(box1.get("x2", 0) - box2.get("x2", 0)) > threshold or
            abs(box1.get("y2", 0) - box2.get("y2", 0)) > threshold):
            return False
        
        # Check if angles are similar (considering angle wrapping)
        angle1 = box1.get("angle", 0) % 180
        angle2 = box2.get("angle", 0) % 180
        angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
        if angle_diff > 10:  # 10 degrees threshold for angles
            return False
        
        return True

    def visualize_detections(self, image_path=None, image=None, spots=None):
        """Visualize detected parking spots
        
        Args:
            image_path: Path to the image
            image: Image as numpy array
            spots: Detected spots (if None, will run detection)
            
        Returns:
            Image with visualized detections
        """
        # Load image if path provided
        if image_path and image is None:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image from {image_path}")
                return None
        
        if image is None:
            logger.error("No image provided for visualization")
            return None
        
        # Run detection if spots not provided
        if spots is None:
            spots_data = self.detect_parking_spots(image=image)
            if spots_data is None:
                logger.error("Failed to detect spots for visualization")
                return None
            spots = spots_data.get("boxes", [])
        elif isinstance(spots, dict):
            # If spots is a dictionary (like the output of detect_parking_spots)
            spots = spots.get("boxes", [])
        
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Draw each spot
        for i, spot in enumerate(spots):
            # Get coordinates
            x1 = spot.get("x1", 0) * img_width
            y1 = spot.get("y1", 0) * img_height
            x2 = spot.get("x2", 0) * img_width
            y2 = spot.get("y2", 0) * img_height
            angle = spot.get("angle", 0)
            
            # Calculate center, width, height
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Create rotated rectangle
            rect = ((center_x, center_y), (width, height), angle)
            box_points = cv2.boxPoints(rect)
            box_points = np.intp(box_points)
            
            # Get group ID and determine color
            group_id = spot.get("group_id", "")
            
            # Generate color based on group_id
            if group_id:
                # Extract group number
                try:
                    group_num = int(group_id.split('_')[1])
                    # Use different colors for different groups
                    colors = [
                        (0, 255, 0),    # Green
                        (0, 0, 255),    # Red
                        (255, 0, 0),    # Blue
                        (0, 255, 255),  # Yellow
                        (255, 0, 255),  # Magenta
                        (255, 255, 0)   # Cyan
                    ]
                    color = colors[group_num % len(colors)]
                except:
                    color = (0, 255, 0)  # Default to green
            else:
                color = (0, 255, 0)  # Default to green
            
            # Draw the rotated rectangle
            cv2.drawContours(vis_image, [box_points], 0, color, 2)
            
            # Add spot ID
            cv2.putText(
                vis_image,
                f"#{i+1}",
                (int(center_x), int(center_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return vis_image

    def save_model(self, path=None):
        """Save the model to a file
        
        Args:
            path: Path to save the model (if None, use self.model_path)
            
        Returns:
            Path to the saved model
        """
        if self.model is None:
            logger.error("No model to save")
            return None
        
        try:
            if path is None:
                if self.model_path:
                    # Use the existing model directory
                    path = os.path.join(self.model_path, "best.pt")
                else:
                    # Create a new directory
                    path = "models/parking_spot_detector.pt"
                    os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            self.model.save(path)
            logger.info(f"Model saved to {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None

    def load_model(self, path):
        """Load a model from a file
        
        Args:
            path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model = YOLO(path)
            self.model_path = path
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


def train_model_from_dataset(dataset_dir="dataset", epochs=50, use_obb=False):
    """Train a model from the dataset
    
    Args:
        dataset_dir: Directory containing the dataset
        epochs: Number of training epochs
        use_obb: Whether to use OBB (Oriented Bounding Box) model
        
    Returns:
        Trained model
    """
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory {dataset_dir} not found")
        return None
    
    # Check if images and labels directories exist
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    if not os.path.exists(images_dir):
        logger.error(f"Images directory {images_dir} not found")
        return None
    
    if not os.path.exists(labels_dir):
        logger.error(f"Labels directory {labels_dir} not found")
        return None
    
    # Create and train the model
    model = ParkingSpotAnnotator()
    model_path = model.train(dataset_dir, epochs=epochs, use_obb=use_obb)
    
    if model_path:
        logger.info(f"Model trained successfully and saved to {model_path}")
        return model
    else:
        logger.error("Model training failed")
        return None


def test_model_on_image(model, image_path):
    """Test the model on an image
    
    Args:
        model: ParkingSpotAnnotator model
        image_path: Path to the image
        
    Returns:
        Visualization image with detections
    """
    # Detect parking spots
    spots = model.detect_parking_spots(image_path=image_path)
    
    # Visualize detections
    vis_image = model.visualize_detections(image_path=image_path, spots=spots)
    
    return vis_image, spots


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parking Spot Annotation Model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", type=str, help="Test the model on an image")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--model", type=str, help="Path to a pre-trained model")
    parser.add_argument("--output", type=str, help="Output path for visualization")
    
    args = parser.parse_args()
    
    if args.train:
        logger.info(f"Training model on dataset {args.dataset} for {args.epochs} epochs")
        model = train_model_from_dataset(args.dataset, args.epochs)
        
    elif args.test:
        logger.info(f"Testing model on image {args.test}")
        
        # Load model
        if args.model:
            model = ParkingSpotAnnotator(args.model)
        else:
            # Try to find a trained model
            if os.path.exists("models/parking_spot_detector.pt"):
                model = ParkingSpotAnnotator("models/parking_spot_detector.pt")
            else:
                logger.error("No model specified and no default model found")
                exit(1)
        
        # Test model
        vis_image, spots = test_model_on_image(model, args.test)
        
        # Save or display visualization
        if args.output:
            cv2.imwrite(args.output, vis_image)
            logger.info(f"Visualization saved to {args.output}")
            
            # Also save spots as JSON
            json_path = os.path.splitext(args.output)[0] + ".json"
            with open(json_path, "w") as f:
                json.dump(spots, f, indent=2)
            logger.info(f"Spots saved to {json_path}")
        else:
            # Display image
            cv2.imshow("Parking Spot Detection", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    else:
        logger.info("No action specified. Use --train or --test.")


