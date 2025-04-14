import cv2
import numpy as np
from custom_yolo_loader import CustomYOLOLoader

def test_custom_yolo():
    """Test our custom YOLO loader"""
    print("Testing custom YOLO loader...")
    
    # Create a test image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)  # Draw a green rectangle
    cv2.rectangle(img, (400, 400), (500, 500), (0, 0, 255), -1)  # Draw a red rectangle
    
    # Initialize our custom YOLO loader
    yolo = CustomYOLOLoader("yolov8n.pt")
    
    # Run detection
    detections = yolo.detect(img)
    
    # Print detections
    print(f"Found {len(detections)} detections:")
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, conf, class_id = detection
        class_name = yolo.class_names.get(int(class_id), f"Class {class_id}")
        print(f"  {i+1}. {class_name}: confidence={conf:.2f}, bbox=({x1},{y1},{x2},{y2})")
    
    # Draw detections on image
    result_img = yolo.draw_detections(img, detections)
    
    # Save and display result
    cv2.imwrite("test_detection_result.jpg", result_img)
    print(f"Result saved to test_detection_result.jpg")
    
    # Try to display if running in an environment with GUI
    try:
        cv2.imshow("Test Detection", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Could not display image (no GUI available)")
    
    return len(detections) > 0

def test_with_real_image():
    """Test with a real image if available"""
    import os
    
    # Try to find a test image
    test_images = ["test.jpg", "test_image.jpg", "sample.jpg", "car.jpg"]
    test_image = None
    
    for img_name in test_images:
        if os.path.exists(img_name):
            test_image = img_name
            break
    
    if test_image is None:
        print("No test image found. Creating a sample image...")
        # Create a sample image with some shapes
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        # Draw a road-like background
        cv2.rectangle(img, (0, 0), (640, 640), (120, 120, 120), -1)  # Gray background
        cv2.rectangle(img, (0, 300), (640, 340), (200, 200, 200), -1)  # Road line
        
        # Draw some car-like rectangles
        cv2.rectangle(img, (100, 200), (200, 250), (0, 0, 255), -1)  # Red car
        cv2.rectangle(img, (300, 400), (400, 450), (255, 0, 0), -1)  # Blue car
        cv2.rectangle(img, (500, 150), (600, 200), (0, 255, 255), -1)  # Yellow car
        
        # Save the sample image
        test_image = "sample_test.jpg"
        cv2.imwrite(test_image, img)
        print(f"Created sample image: {test_image}")
    else:
        print(f"Using existing test image: {test_image}")
    
    # Load the image
    img = cv2.imread(test_image)
    if img is None:
        print(f"Failed to load image: {test_image}")
        return False
    
    # Initialize our custom YOLO loader
    yolo = CustomYOLOLoader("yolov8n.pt")
    
    # Run detection
    detections = yolo.detect(img)
    
    # Print detections
    print(f"Found {len(detections)} detections in {test_image}:")
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, conf, class_id = detection
        class_name = yolo.class_names.get(int(class_id), f"Class {class_id}")
        print(f"  {i+1}. {class_name}: confidence={conf:.2f}, bbox=({x1},{y1},{x2},{y2})")
    
    # Draw detections on image
    result_img = yolo.draw_detections(img, detections)
    
    # Save result
    result_path = f"detection_result_{os.path.basename(test_image)}"
    cv2.imwrite(result_path, result_img)
    print(f"Result saved to {result_path}")
    
    # Try to display if running in an environment with GUI
    try:
        cv2.imshow("Real Image Detection", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Could not display image (no GUI available)")
    
    return len(detections) > 0

if __name__ == "__main__":
    print("=== Testing with synthetic image ===")
    synthetic_test_passed = test_custom_yolo()
    
    print("\n=== Testing with real image ===")
    real_test_passed = test_with_real_image()
    
    if synthetic_test_passed and real_test_passed:
        print("\n✅ All tests passed! The custom YOLO loader is working correctly.")
    elif synthetic_test_passed:
        print("\n⚠️ Synthetic test passed but real image test failed.")
    elif real_test_passed:
        print("\n⚠️ Real image test passed but synthetic test failed.")
    else:
        print("\n❌ All tests failed. The custom YOLO loader is not working correctly.")
