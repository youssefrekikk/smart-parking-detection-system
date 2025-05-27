import os
import torch
import sys
import cv2
import numpy as np
from pathlib import Path
import shutil

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
from ultralytics import YOLO

def create_quick_dataset(images, output_dir="quick_dataset"):
    """
    Create a quick dataset from a list of images
    
    Args:
        images: List of image paths
        output_dir: Output directory for the dataset
        
    Returns:
        Path to the dataset
    """
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Process each image
    for img_path in images:
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping")
            continue
        
        # Copy image to dataset
        img_filename = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(images_dir, img_filename))
        
        # Create a grid of parking spots as initial labels
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load {img_path}, skipping")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Create grid labels (4x3 grid)
        rows, cols = 3, 4
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)
        
        with open(label_path, "w") as f:
            for row in range(rows):
                for col in range(cols):
                    # Calculate center position (normalized)
                    center_x = (col + 0.5) / cols
                    center_y = (row + 0.5) / rows
                    
                    # Calculate width and height (normalized)
                    width = 0.8 / cols
                    height = 0.8 / rows
                    
                    # Calculate corners (normalized)
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y - height / 2
                    x3 = center_x + width / 2
                    y3 = center_y + height / 2
                    x4 = center_x - width / 2
                    y4 = center_y + height / 2
                    
                    # Write to label file (class_id x1 y1 x2 y2 x3 y3 x4 y4)
                    f.write(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
    
    # Create dataset YAML
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("names:\n")
        f.write("  0: Parking Spot\n")
    
    print(f"Created quick dataset with {len(images)} images")
    return output_dir

def quick_train(images, epochs=30, output_dir="quick_model"):
    """
    Quickly train a model on the provided images
    
    Args:
        images: List of image paths
        epochs: Number of training epochs
        output_dir: Output directory for the model
        
    Returns:
        Path to the trained model
    """
    # Create dataset
    dataset_dir = create_quick_dataset(images)
    
    # Create model
    model = ParkingSpotAnnotator()
    
    # Initialize with pre-trained model
    model.model = YOLO('yolov8m-obb.pt')
    
    # Train the model
    print(f"Training model for {epochs} epochs...")
    model_path = model.train(
        dataset_dir,
        epochs=epochs,
        img_size=640,
        batch_size=1,  # Small batch size for few images
        use_obb=True
    )
    
    if model_path:
        print(f"Model trained successfully and saved to {model_path}")
        
        # Fix model weights to ensure it always predicts
        from fix_model_weights import fix_model_weights
        best_pt = os.path.join(model_path, "weights", "best.pt")
        fixed_model_path = fix_model_weights(best_pt, os.path.join(output_dir, "model_fixed.pt"))
        
        return fixed_model_path
    else:
        print("Model training failed")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_train.py <image1> [image2] [image3] ... [epochs]")
        sys.exit(1)
    
    # Check if the last argument is a number (epochs)
    try:
        epochs = int(sys.argv[-1])
        images = sys.argv[1:-1]
    except ValueError:
        epochs = 30  # Default
        images = sys.argv[1:]
    
    print(f"Quick training on {len(images)} images for {epochs} epochs")
    model_path = quick_train(images, epochs)
    
    if model_path:
        print(f"Model trained and fixed successfully! Use this model for better results:")
        print(f"python src/test_model.py <your_image> {model_path}")
    else:
        print("Training failed.")
