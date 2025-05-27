import os
import cv2
import numpy as np
import random
import shutil

def augment_dataset(source_dir="dataset_regular", output_dir="dataset_augmented", augmentation_factor=10):
    """
    Create an augmented dataset from the converted regular YOLO dataset
    
    Args:
        source_dir: Source dataset directory (should be dataset_regular)
        output_dir: Output directory for augmented dataset
        augmentation_factor: Number of augmented images to create per original image
        
    Returns:
        Path to the augmented dataset
    """
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
    source_images_dir = os.path.join(source_dir, "images")
    source_labels_dir = os.path.join(source_dir, "labels")
    
    if not os.path.exists(source_images_dir):
        print(f"Error: {source_images_dir} does not exist!")
        return None
    
    image_files = [f for f in os.listdir(source_images_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} original images in {source_dir}")
    
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(source_images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(source_labels_dir, label_file)
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"Warning: No label file for {img_file}, skipping")
            continue
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load {img_path}, skipping")
            continue
        
        # Load labels (regular YOLO format: class center_x center_y width height)
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
            aug_labels = label_lines.copy()
            h, w = img.shape[:2]
            
            # Randomly select augmentation types
            aug_types = random.sample([
                'brightness', 'contrast', 'blur', 'noise', 
                'flip', 'rotate', 'scale'
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
                
                # Adjust labels for horizontal flip
                flipped_labels = []
                for line in aug_labels:
                    parts = line.strip().split()
                    if len(parts) == 5:  # class center_x center_y width height
                        class_id = parts[0]
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Flip x coordinate (1 - center_x)
                        center_x = 1.0 - center_x
                        
                        # Reconstruct the line
                        flipped_line = f"{class_id} {center_x} {center_y} {width} {height}\n"
                        flipped_labels.append(flipped_line)
                
                aug_labels = flipped_labels
            
            if 'rotate' in aug_types:
                # Mild rotation (-10 to 10 degrees)
                angle = random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                aug_img = cv2.warpAffine(aug_img, M, (w, h))
                
                # For mild rotations, we keep the original labels
                # For more complex rotations, you'd need to transform the coordinates
            
            if 'scale' in aug_types:
                # Random scaling
                scale = random.uniform(0.9, 1.1)
                M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
                aug_img = cv2.warpAffine(aug_img, M, (w, h))
                
                # For mild scaling, we keep the original labels
            
            # Save augmented image
            cv2.imwrite(os.path.join(images_dir, aug_img_file), aug_img)
            
            # Save augmented labels
            with open(os.path.join(labels_dir, aug_label_file), 'w') as f:
                f.writelines(aug_labels)
    
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

if __name__ == "__main__":
    import sys
    
    source_dir = "dataset_regular"  # Default to converted dataset
    augmentation_factor = 10  # Default
    
    if len(sys.argv) > 1:
        try:
            augmentation_factor = int(sys.argv[1])
        except:
            pass
    
    if len(sys.argv) > 2:
        source_dir = sys.argv[2]
    
    print(f"Creating augmented dataset from {source_dir} with factor {augmentation_factor}")
    augment_dataset(source_dir, "dataset_augmented", augmentation_factor)
