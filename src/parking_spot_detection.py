import cv2
import numpy as np
import json
from ultralytics import YOLO
import os
os.environ["TORCH_ALLOW_WEIGHTS_ONLY_SKIP"] = "1"
import torch
import torch.serialization
def load_parking_spots_from_json(json_path, frame_width, frame_height):
    """
    Load parking spots from our JSON mask format
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if it's our enhanced format with camera_id, boxes, groups
    if isinstance(data, dict) and 'boxes' in data:
        boxes = data['boxes']
        # We can also access groups if needed
        groups = data.get('groups', [])
    else:
        boxes = data  # Simple array format
    
    parking_spots = []
    
    for i, box in enumerate(boxes):
        # Check if we have the format with x1, y1, x2, y2
        if 'x1' in box and 'y1' in box and 'x2' in box and 'y2' in box:
            # Format with x1, y1, x2, y2
            x1 = box.get('x1', 0)
            y1 = box.get('y1', 0)
            x2 = box.get('x2', 0)
            y2 = box.get('y2', 0)
            angle = box.get('angle', 0)
            
            # Convert to pixel coordinates
            x1_px = int(x1 * frame_width)
            y1_px = int(y1 * frame_height)
            x2_px = int(x2 * frame_width)
            y2_px = int(y2 * frame_height)
            
            # Calculate center, width, height
            x_center_px = (x1_px + x2_px) / 2
            y_center_px = (y1_px + y2_px) / 2
            width_px = abs(x2_px - x1_px)
            height_px = abs(y2_px - y1_px)
            
            # Create corners for the rotated rectangle
            rect = ((x_center_px, y_center_px), (width_px, height_px), angle)
            box_points = cv2.boxPoints(rect)
            
            # Extract group information
            group_id = box.get('groupId') or box.get('group_id')
            location = box.get('location')
            
        # Check if we have the format with class_id, x_center, y_center
        elif 'class_id' in box and 'x_center' in box and 'width' in box:
            # Format from the logs
            x_center = box.get('x_center', 0)
            y_center = box.get('y_center', 0)
            width = box.get('width', 0)
            height = box.get('height', 0)
            angle = box.get('angle', 0)
            
            # Convert to pixel coordinates
            x_center_px = int(x_center * frame_width)
            y_center_px = int(y_center * frame_height)
            width_px = int(width * frame_width)
            height_px = int(height * frame_height)
            
            # Calculate corners
            rect = ((x_center_px, y_center_px), (width_px, height_px), angle)
            box_points = cv2.boxPoints(rect)
            
            # Extract group information
            group_id = box.get('group_id')
            location = box.get('location')
        else:
            print(f"Warning: Box {i} has an unsupported format: {box}")
            continue
        
        # Convert box_points to a numpy array to ensure it's properly formatted
        box_points = np.array(box_points)
        
        # Add to parking spots
        spot = {
            "id": i + 1,
            "x_center": int(x_center_px),
            "y_center": int(y_center_px),
            "w": int(width_px),
            "h": int(height_px),
            "angle": angle,
            "corners": box_points,
            "group_id": group_id,
            "location": location
        }
        
        parking_spots.append(spot)
    
    # If we have group information in the enhanced format, we can use it to update spot locations
    if isinstance(data, dict) and 'groups' in data:
        groups = data['groups']
        for group in groups:
            group_id = group.get('group_id')
            location = group.get('location')
            box_indices = group.get('box_indices', [])
            
            # Update spots with this group's information
            for idx in box_indices:
                if 0 <= idx < len(parking_spots):
                    parking_spots[idx]['group_id'] = group_id
                    if location:
                        parking_spots[idx]['location'] = location
    
    return parking_spots

def draw_rotated_box(frame, corners, color, thickness=2):
    """
    Draw a rotated bounding box on the frame using its 4 corner points.
    """
    corners = np.intp(corners)
    cv2.polylines(frame, [corners], isClosed=True, color=color, thickness=thickness)

def calculate_box_angle(corners):
    """Calculate angle of the box from its corners"""
    edge = corners[1] - corners[0]
    angle = np.arctan2(edge[1], edge[0])
    return np.degrees(angle)

def obb_intersection_area(boxA, boxB):
    """
    Calculate intersection area between rotated boxes
    """
    try:
        # Convert inputs to proper format to avoid slice issues
        # boxA and boxB should be tuples of (x_coords, y_coords)
        boxA_x = np.array(boxA[0], dtype=np.float32)
        boxA_y = np.array(boxA[1], dtype=np.float32)
        boxB_x = np.array(boxB[0], dtype=np.float32)
        boxB_y = np.array(boxB[1], dtype=np.float32)
        
        # Create masks for both boxes
        min_x = int(min(np.min(boxA_x), np.min(boxB_x)))
        min_y = int(min(np.min(boxA_y), np.min(boxB_y)))
        max_x = int(max(np.max(boxA_x), np.max(boxB_x)))
        max_y = int(max(np.max(boxA_y), np.max(boxB_y)))
        
        # Ensure we have a valid mask size
        w = max(1, max_x - min_x + 1)
        h = max(1, max_y - min_y + 1)
        
        # Create masks
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask2 = np.zeros((h, w), dtype=np.uint8)
        
        # Create points for fillPoly
        box1_points = np.array([
            [int(x - min_x), int(y - min_y)] 
            for x, y in zip(boxA_x, boxA_y)
        ], dtype=np.int32)
        
        box2_points = np.array([
            [int(x - min_x), int(y - min_y)] 
            for x, y in zip(boxB_x, boxB_y)
        ], dtype=np.int32)
        
        # Ensure points are properly formatted for fillPoly
        box1_points = box1_points.reshape((-1, 1, 2))
        box2_points = box2_points.reshape((-1, 1, 2))
        
        # Fill polygons
        cv2.fillPoly(mask1, [box1_points], 1)
        cv2.fillPoly(mask2, [box2_points], 1)
        
        # Calculate intersection
        intersection = cv2.bitwise_and(mask1, mask2)
        return cv2.countNonZero(intersection)
    
    except Exception as e:
        print(f"Error in obb_intersection_area: {e}")
        import traceback
        print(traceback.format_exc())
        return 0

def detect_parking_spots(frame, parking_spots, model_path="yolov8n.pt", overlap_threshold=0.4):
    """
    Detect available parking spots in a frame using YOLOv8
    """
    try:
        # Try to load the model with a direct approach for PyTorch 2.6+
        try:
            # Monkey patch torch.load to use weights_only=False
            original_torch_load = torch.load
            
            def patched_torch_load(f, *args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(f, *args, **kwargs)
            
            # Replace torch.load temporarily
            torch.load = patched_torch_load
            
            # Now try to load the model
            model = YOLO(model_path)
            
            # Restore original torch.load
            torch.load = original_torch_load
            
            print("Successfully loaded YOLO model with patched torch.load")
        except Exception as e:
            print(f"Error loading YOLO with patched method: {e}")
            # Fall back to simple detection
            return fallback_detection(frame, parking_spots)
        
        # Initialize spot status
        spot_status = {spot["id"]: "Empty" for spot in parking_spots}
        
        # Run YOLOv8 detection
        results = model(frame)
        
        # Process detections
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls].lower()
                
                # We're only interested in vehicles (car, truck, bus, motorcycle, bicycle)
                if label not in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                    continue
                
                # Create detection box corners
                det_corners = np.array([
                    [x1, y1], [x2, y1],
                    [x2, y2], [x1, y2]
                ])
                
                # Check each parking spot
                for spot in parking_spots:
                    try:
                        # Extract corners as separate x and y arrays
                        spot_corners = spot["corners"]
                        spot_x = spot_corners[:, 0].astype(np.float32)
                        spot_y = spot_corners[:, 1].astype(np.float32)
                        
                        # Create detection box corners as separate x and y arrays
                        det_x = det_corners[:, 0].astype(np.float32)
                        det_y = det_corners[:, 1].astype(np.float32)
                        
                        # Pass as tuples to avoid slice issues
                        det_box = (det_x, det_y)
                        spot_box = (spot_x, spot_y)
                        
                        inter_area = obb_intersection_area(det_box, spot_box)
                        spot_area = spot["w"] * spot["h"]
                        
                        if spot_area > 0:
                            overlap_ratio = inter_area / float(spot_area)
                            if overlap_ratio >= overlap_threshold:
                                spot_status[spot["id"]] = "Filled"
                    except Exception as e:
                        print(f"Error checking spot {spot['id']}: {e}")
                        import traceback
                        print(traceback.format_exc())
        
        # Create visualization frame
        vis_frame = frame.copy()
        
        # Draw detections
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls].lower()
                
                if label in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_frame, 
                               f"{label} {conf:.2f}", 
                               (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, 
                               (0, 255, 0), 
                               2)
        
        # Draw parking spots
        for spot in parking_spots:
            color = (0, 255, 0) if spot_status[spot["id"]] == "Empty" else (0, 0, 255)
            draw_rotated_box(vis_frame, spot["corners"], color, 2)
            
            text_pos = (int(spot["x_center"]) - 20, int(spot["y_center"]) - 10)
            cv2.putText(vis_frame,
                       f"Spot {spot['id']}",
                       text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       color,
                       2)
        
        # Calculate statistics
        total_spots = len(parking_spots)
        available_spots = sum(1 for status in spot_status.values() if status == "Empty")
        
        # Group statistics
        group_stats = {}
        # Group spots by group_id
        grouped_spots = {}
        for spot in parking_spots:
            group_id = spot.get("group_id")
            if group_id:
                if group_id not in grouped_spots:
                    grouped_spots[group_id] = []
                grouped_spots[group_id].append(spot)
        
        # Calculate stats for each group
        for group_id, spots in grouped_spots.items():
            group_total = len(spots)
            group_available = sum(1 for spot in spots if spot_status[spot["id"]] == "Empty")
            location = spots[0].get("location", "Unknown")
            
            group_stats[group_id] = {
                "location": location,
                "total": group_total,
                "available": group_available,
                "percentage": round(group_available / max(group_total, 1) * 100, 1)
            }
        
        # Add counter text to the visualization
        counter_text = f"Available: {available_spots}/{total_spots}"
        text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        
        # Position in bottom right corner with padding
        text_x = vis_frame.shape[1] - text_size[0] - 20
        text_y = vis_frame.shape[0] - 20
        
        # Draw background rectangle for better visibility
        cv2.rectangle(vis_frame, 
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0),
                    -1)
        
        # Draw counter text
        cv2.putText(vis_frame,
                    counter_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)
        
        return {
            "total_spots": total_spots,
            "available_spots": available_spots,
            "percentage_available": round(available_spots / max(total_spots, 1) * 100, 1),
            "spot_status": spot_status,
            "group_stats": group_stats,
            "visualization_frame": vis_frame
        }
        
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        # Fallback to a simpler detection if YOLO fails
        return fallback_detection(frame, parking_spots)
def fallback_detection(frame, parking_spots):
    """
    Simple fallback detection when YOLO is not available
    Uses basic motion detection and color analysis
    """
    # Convert frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Initialize spot status
    spot_status = {spot["id"]: "Empty" for spot in parking_spots}
    
    # Check each parking spot
    for spot in parking_spots:
        # Extract the region of interest (ROI) for this spot
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        corners = np.intp(spot["corners"])
        cv2.fillPoly(mask, [corners], 255)
        
        # Apply mask to get the spot region
        spot_roi = cv2.bitwise_and(blurred, blurred, mask=mask)
        
        # Calculate the average intensity in the spot
        if np.sum(mask) > 0:  # Avoid division by zero
            avg_intensity = np.sum(spot_roi) / np.sum(mask)
        else:
            avg_intensity = 0
        
        # Calculate the standard deviation of intensity in the spot
        # Higher std_dev usually means there's an object (car) in the spot
        non_zero_vals = spot_roi[mask > 0]
        if len(non_zero_vals) > 0:
            std_dev = np.std(non_zero_vals)
        else:
            std_dev = 0
        
        # Simple heuristic: if std_dev is high enough, consider the spot filled
        # This threshold may need adjustment based on lighting conditions
        if std_dev > 25:  # Arbitrary threshold
            spot_status[spot["id"]] = "Filled"
    
    # Create visualization frame
    vis_frame = frame.copy()
    
    # Draw parking spots
    for spot in parking_spots:
        color = (0, 255, 0) if spot_status[spot["id"]] == "Empty" else (0, 0, 255)
        draw_rotated_box(vis_frame, spot["corners"], color, 2)
        
        text_pos = (int(spot["x_center"]) - 20, int(spot["y_center"]) - 10)
        cv2.putText(vis_frame,
                   f"Spot {spot['id']}",
                   text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   color,
                   2)
    
    # Calculate statistics
    total_spots = len(parking_spots)
    available_spots = sum(1 for status in spot_status.values() if status == "Empty")
    
    # Group statistics
    group_stats = {}
    # Group spots by group_id
    grouped_spots = {}
    for spot in parking_spots:
        group_id = spot.get("group_id")
        if group_id:
            if group_id not in grouped_spots:
                grouped_spots[group_id] = []
            grouped_spots[group_id].append(spot)
    
    # Calculate stats for each group
    for group_id, spots in grouped_spots.items():
        group_total = len(spots)
        group_available = sum(1 for spot in spots if spot_status[spot["id"]] == "Empty")
        location = spots[0].get("location", "Unknown")
        
        group_stats[group_id] = {
            "location": location,
            "total": group_total,
            "available": group_available,
            "percentage": round(group_available / max(group_total, 1) * 100, 1)
        }
    
    return {
        "total_spots": total_spots,
        "available_spots": available_spots,
        "percentage_available": round(available_spots / max(total_spots, 1) * 100, 1),
        "spot_status": spot_status,
        "group_stats": group_stats,
        "visualization_frame": vis_frame
    }

def test_detection(mask_path, image_path, output_path=None, model_path="yolov8n.pt"):
    """
    Test the parking spot detection on a single image
    """
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Get image dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Load parking spots
    try:
        parking_spots = load_parking_spots_from_json(mask_path, frame_width, frame_height)
        print(f"Loaded {len(parking_spots)} parking spots from {mask_path}")
    except Exception as e:
        print(f"Error loading parking spots: {e}")
        return
    
    # Run detection
    try:
        results = detect_parking_spots(frame, parking_spots, model_path)
    except Exception as e:
        print(f"Error in detection: {e}")
        # Try fallback
        results = fallback_detection(frame, parking_spots)
    
    # Print results
    print(f"Total spots: {results['total_spots']}")
    print(f"Available spots: {results['available_spots']}")
    print(f"Percentage available: {results['percentage_available']}%")
    
    # Print group stats if available
    if results['group_stats']:
        print("\nGroup Statistics:")
        for group_id, stats in results['group_stats'].items():
            print(f"  Group {stats['location']}: {stats['available']}/{stats['total']} available ({stats['percentage']}%)")
    
    # Save or display visualization
    if output_path:
        cv2.imwrite(output_path, results['visualization_frame'])
        print(f"Saved visualization to {output_path}")
    else:
        cv2.imshow("Parking Detection", results['visualization_frame'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) >= 3:
        mask_path = sys.argv[1]
        image_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        model_path = sys.argv[4] if len(sys.argv) > 4 else "yolov8n.pt"
        test_detection(mask_path, image_path, output_path, model_path)
    else:
        print("Usage: python parkingspot_detection.py <mask_path> <image_path> [output_path] [model_path]")
