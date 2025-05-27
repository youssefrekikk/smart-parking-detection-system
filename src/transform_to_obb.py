import cv2
import numpy as np
import math

def transform_detections_to_obb(detections, image, method="contour_analysis"):
    """
    Transform regular YOLO detections to OBB format
    
    Args:
        detections: Regular YOLO detections (x1, y1, x2, y2 format)
        image: Original image
        method: Method to use for OBB transformation
        
    Returns:
        OBB detections with rotation and corners
    """
    obb_detections = []
    
    for detection in detections:
        x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        
        # Convert normalized coordinates to pixel coordinates
        img_h, img_w = image.shape[:2]
        x1_px = int(x1 * img_w)
        y1_px = int(y1 * img_h)
        x2_px = int(x2 * img_w)
        y2_px = int(y2 * img_h)
        
        # Ensure coordinates are within image bounds
        x1_px = max(0, min(x1_px, img_w - 1))
        y1_px = max(0, min(y1_px, img_h - 1))
        x2_px = max(0, min(x2_px, img_w - 1))
        y2_px = max(0, min(y2_px, img_h - 1))
        
        # Extract the region of interest
        roi = image[y1_px:y2_px, x1_px:x2_px]
        
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            # Fallback to rectangular box
            angle = 0
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        else:
            if method == "contour_analysis":
                angle, corners = _analyze_contours(roi, x1, y1, x2, y2)
            elif method == "edge_detection":
                angle, corners = _analyze_edges(roi, x1, y1, x2, y2)
            elif method == "hough_lines":
                angle, corners = _analyze_hough_lines(roi, x1, y1, x2, y2)
            else:
                # Default rectangular
                angle = 0
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        
        # Create OBB detection
        obb_detection = detection.copy()
        obb_detection['angle'] = angle
        obb_detection['corners'] = corners.tolist()
        
        obb_detections.append(obb_detection)
    
    return obb_detections

def _analyze_contours(roi, x1, y1, x2, y2):
    """Analyze contours to find orientation"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Fit minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            center, (width, height), angle = rect
            
            # Get box points
            box_points = cv2.boxPoints(rect)
            
            # Convert back to normalized coordinates
            roi_h, roi_w = roi.shape[:2]
            normalized_corners = []
            for point in box_points:
                norm_x = x1 + (point[0] / roi_w) * (x2 - x1)
                norm_y = y1 + (point[1] / roi_h) * (y2 - y1)
                normalized_corners.append([norm_x, norm_y])
            
            return angle, np.array(normalized_corners)
        
    except Exception as e:
        pass
    
    # Fallback to rectangular
    return 0, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

def _analyze_edges(roi, x1, y1, x2, y2):
    """Analyze edges to find orientation"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Combine all contours
            all_points = np.vstack(contours)
            
            # Fit minimum area rectangle
            rect = cv2.minAreaRect(all_points)
            center, (width, height), angle = rect
            
            # Get box points
            box_points = cv2.boxPoints(rect)
            
            # Convert back to normalized coordinates
            roi_h, roi_w = roi.shape[:2]
            normalized_corners = []
            for point in box_points:
                norm_x = x1 + (point[0] / roi_w) * (x2 - x1)
                norm_y = y1 + (point[1] / roi_h) * (y2 - y1)
                normalized_corners.append([norm_x, norm_y])
            
            return angle, np.array(normalized_corners)
        
    except Exception as e:
        pass
    
    # Fallback to rectangular
    return 0, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

def _analyze_hough_lines(roi, x1, y1, x2, y2):
    """Analyze Hough lines to find dominant orientation"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
        
        if lines is not None:
            # Find dominant angle
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                angles.append(angle)
            
            # Get most common angle
            dominant_angle = np.median(angles)
            
            # Create rotated rectangle with dominant angle
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Calculate corners of rotated rectangle
            corners = _get_rotated_corners(center_x, center_y, width, height, dominant_angle)
            
            return dominant_angle, corners
        
    except Exception as e:
        pass
    
    # Fallback to rectangular
    return 0, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

def _get_rotated_corners(center_x, center_y, width, height, angle_deg):
    """Calculate corners of a rotated rectangle"""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Half dimensions
    hw = width / 2
    hh = height / 2
    
    # Corner offsets
    corners_offset = [
        (-hw, -hh),
        (hw, -hh),
        (hw, hh),
        (-hw, hh)
    ]
    
    # Rotate and translate corners
    corners = []
    for dx, dy in corners_offset:
        x = center_x + dx * cos_a - dy * sin_a
        y = center_y + dx * sin_a + dy * cos_a
        corners.append([x, y])
    
    return np.array(corners)
