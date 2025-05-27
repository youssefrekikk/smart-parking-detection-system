import os
import torch
import sys
import yaml
import random
import cv2
import numpy as np
from pathlib import Path

# Set environment variable
os.environ["TORCH_ALLOW_WEIGHTS_ONLY_SKIP"] = "1"

# Patch torch.load to use weights_only=False
original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(f, *args, **kwargs)

# Replace torch.load with our patched version
torch.load = patched_torch_load

# Patch numpy to provide deprecated types for older libraries
if not hasattr(np, 'bool'):
    np.bool = bool

if not hasattr(np, 'complex'):
    np.complex = complex

if not hasattr(np, 'float'):
    np.float = float

if not hasattr(np, 'int'):
    np.int = int

if not hasattr(np, 'object'):
    np.object = object

if not hasattr(np, 'str'):
    np.str = str
    
# Import the model after patching
from parking_spot_annotation_model import ParkingSpotAnnotator

def create_augmented_dataset(source_dir, output_dir="dataset_augmented", augmentation_factor=10):
    """
    Create an augmented dataset from the original dataset
    
    Args:
        source_dir: Original dataset directory
        output_dir: Output directory for augmented dataset
        augmentation_factor: Number of augmented images to create per original image
        
    Returns:
        Path to the augmented dataset
    """
    import shutil
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Copy notes.json if it exists
    notes_path = os.path.join(source_dir, "notes.json")
    if os.path.exists(notes_path):
        shutil.copy(notes_path, os.path.join(output_dir, "notes.json"))
    
    # Get all image files
    image_files = [f for f in os.listdir(os.path.join(source_dir, "images")) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} original images")
    
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(source_dir, "images", img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(source_dir, "labels", label_file)
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"Warning: No label file for {img_file}, skipping")
            continue
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load {img_path}, skipping")
            continue
        
        # Load label (YOLO OBB format)
        with open(label_path, 'r') as f:
            label_lines = f.readlines()
        
        # Copy original image and label
        shutil.copy(img_path, os.path.join(images_dir, img_file))
        shutil.copy(label_path, os.path.join(labels_dir, label_file))
        
        # Create augmented versions
        for i in range(augmentation_factor):
            # Generate unique filename for augmented image
            aug_img_file = f"{os.path.splitext(img_file)[0]}_aug{i}{os.path.splitext(img_file)[1]}"
            aug_label_file = f"{os.path.splitext(img_file)[0]}_aug{i}.txt"
            
            # Apply augmentations
            aug_img = img.copy()
            h, w = img.shape[:2]
            
            # Randomly select augmentation types
            aug_types = random.sample([
                'brightness', 'contrast', 'blur', 'noise', 
                'flip', 'rotate', 'scale', 'translate'
            ], k=random.randint(1, 4))
            
            # Apply selected augmentations
            if 'brightness' in aug_types:
                # Brightness adjustment
                beta = random.uniform(-30, 30)
                aug_img = cv2.convertScaleAbs(aug_img, beta=beta)
            
            if 'contrast' in aug_types:
                # Contrast adjustment
                alpha = random.uniform(0.8, 1.2)
                aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha)
            
            if 'blur' in aug_types:
                # Gaussian blur
                blur_size = random.choice([3, 5, 7])
                aug_img = cv2.GaussianBlur(aug_img, (blur_size, blur_size), 0)
            
            if 'noise' in aug_types:
                # Add random noise
                noise = np.random.normal(0, random.uniform(5, 15), aug_img.shape).astype(np.uint8)
                aug_img = cv2.add(aug_img, noise)
            
            if 'flip' in aug_types and random.random() > 0.5:
                # Horizontal flip
                aug_img = cv2.flip(aug_img, 1)
                
                # Need to adjust labels for horizontal flip
                flipped_labels = []
                for line in label_lines:
                    parts = line.strip().split()
                    class_id = parts[0]
                    coords = [float(x) for x in parts[1:]]
                    
                    # For OBB format (class_id x1 y1 x2 y2 x3 y3 x4 y4)
                    # Flip x coordinates (1-x)
                    for i in range(0, len(coords), 2):
                        coords[i] = 1.0 - coords[i]
                    
                    # Reorder points to maintain correct orientation
                    if len(coords) == 8:  # 4 points (x,y)
                        # Reorder points: swap 0<->3 and 1<->2
                        x1, y1 = coords[0], coords[1]
                        x2, y2 = coords[2], coords[3]
                        x3, y3 = coords[4], coords[5]
                        x4, y4 = coords[6], coords[7]
                        
                        # New order after flip
                        coords = [x4, y4, x3, y3, x2, y2, x1, y1]
                    
                    # Reconstruct the line
                    flipped_line = class_id + " " + " ".join([str(c) for c in coords]) + "\n"
                    flipped_labels.append(flipped_line)
                
                label_lines = flipped_labels
            
            if 'rotate' in aug_types:
                # Mild rotation (-10 to 10 degrees)
                angle = random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                aug_img = cv2.warpAffine(aug_img, M, (w, h))
                
                # For simplicity, we'll keep the original labels for mild rotations
            
            if 'scale' in aug_types:
                # Random scaling
                scale = random.uniform(0.9, 1.1)
                M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
                aug_img = cv2.warpAffine(aug_img, M, (w, h))
                
                # For simplicity, we'll keep the original labels for mild scaling
            
            if 'translate' in aug_types:
                # Random translation
                tx = random.uniform(-w*0.1, w*0.1)
                ty = random.uniform(-h*0.1, h*0.1)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                aug_img = cv2.warpAffine(aug_img, M, (w, h))
                
                # For simplicity, we'll keep the original labels for mild translations
            
            # Save augmented image
            cv2.imwrite(os.path.join(images_dir, aug_img_file), aug_img)
            
            # Save augmented labels
            with open(os.path.join(labels_dir, aug_label_file), 'w') as f:
                f.writelines(label_lines)
    
    # Create dataset YAML
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    
    # Load class names from notes.json
    if os.path.exists(os.path.join(output_dir, "notes.json")):
        with open(os.path.join(output_dir, "notes.json"), 'r') as f:
            import json
            notes = json.load(f)
        
        # Extract class names
        names = [category['name'] for category in notes.get('categories', [])]
    else:
        # Default class name if notes.json is not available
        names = ['Parking Spot']
    
    # Create dataset YAML content
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n")
        f.write("names:\n")
        for i, name in enumerate(names):
            f.write(f"  {i}: {name}\n")
    
    print(f"Created augmented dataset with {len(image_files) * (augmentation_factor + 1)} images")
    return output_dir

def train_with_augmentation(source_dir="dataset", epochs=100, augmentation_factor=10):
    """
    Train a model with data augmentation
    
    Args:
        source_dir: Original dataset directory
        epochs: Number of training epochs
        augmentation_factor: Number of augmented images to create per original image
        
    Returns:
        Trained model
    """
    # Create augmented dataset
    augmented_dir = create_augmented_dataset(source_dir, augmentation_factor=augmentation_factor)
    
    # Create model
    model = ParkingSpotAnnotator()
    
    # Additional training hyperparameters
    hyperparams = {
        'img_size': 640,
        'batch_size': 8,  # Smaller batch size for better generalization
        'patience': 20,   # Increased patience for early stopping
        'lr0': 0.01,      # Initial learning rate
        'lrf': 0.01,      # Final learning rate as a fraction of lr0
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 0.05,      # Box loss gain
        'cls': 0.5,       # Class loss gain
        'hsv_h': 0.015,   # Image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,     # Image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,     # Image HSV-Value augmentation (fraction)
        'degrees': 0.0,   # Image rotation (+/- deg)
        'translate': 0.1, # Image translation (+/- fraction)
        'scale': 0.5,     # Image scale (+/- gain)
        'shear': 0.0,     # Image shear (+/- deg)
        'perspective': 0.0, # Image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,    # Image flip up-down (probability)
        'fliplr': 0.5,    # Image flip left-right (probability)
        'mosaic': 1.0,    # Image mosaic (probability)
        'mixup': 0.0,     # Image mixup (probability)
        'copy_paste': 0.0 # Segment copy-paste (probability)
    }
    
    # Train the model
    model_path = model.train(
        augmented_dir, 
        epochs=epochs, 
        img_size=hyperparams['img_size'],
        batch_size=hyperparams['batch_size'],
        use_obb=True
    )
    
    if model_path:
        print(f"Model trained successfully and saved to {model_path}")
        return model
    else:
        print("Model training failed")
        return None

if __name__ == "__main__":
    # Parse command line arguments
    epochs = 100  # Default to more epochs for small dataset
    augmentation_factor = 10  # Default augmentation factor
    
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except:
            pass
    
    if len(sys.argv) > 2:
        try:
            augmentation_factor = int(sys.argv[2])
        except:
            pass
    
    print(f"Training with {epochs} epochs and augmentation factor {augmentation_factor}")
    
    # Train the model
    model = train_with_augmentation("dataset", epochs=epochs, augmentation_factor=augmentation_factor)
    
    if model:
        print(f"Model trained successfully!")
    else:
        print(f"Model training failed.")
