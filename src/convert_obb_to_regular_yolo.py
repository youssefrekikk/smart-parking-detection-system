import os
import numpy as np
import shutil

def convert_obb_to_regular_yolo(source_dir="dataset", output_dir="dataset_regular"):
    """
    Convert OBB format labels to regular YOLO format for training
    This preserves the bounding area but loses rotation info
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    # Copy images
    source_images = os.path.join(source_dir, "images")
    output_images = os.path.join(output_dir, "images")
    
    for img_file in os.listdir(source_images):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy(os.path.join(source_images, img_file), 
                       os.path.join(output_images, img_file))
    
    # Convert labels
    source_labels = os.path.join(source_dir, "labels")
    output_labels = os.path.join(output_dir, "labels")
    
    for label_file in os.listdir(source_labels):
        if label_file.endswith('.txt'):
            input_path = os.path.join(source_labels, label_file)
            output_path = os.path.join(output_labels, label_file)
            
            with open(input_path, 'r') as f:
                lines = f.readlines()
            
            converted_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 9:  # class + 8 coordinates (x1 y1 x2 y2 x3 y3 x4 y4)
                    class_id = parts[0]
                    coords = [float(x) for x in parts[1:9]]
                    
                    # Extract x and y coordinates
                    x_coords = coords[0::2]  # x1, x2, x3, x4
                    y_coords = coords[1::2]  # y1, y2, y3, y4
                    
                    # Find bounding box
                    x_min = min(x_coords)
                    x_max = max(x_coords)
                    y_min = min(y_coords)
                    y_max = max(y_coords)
                    
                    # Convert to YOLO format (center_x, center_y, width, height)
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    converted_lines.append(f"{class_id} {center_x} {center_y} {width} {height}\n")
            
            with open(output_path, 'w') as f:
                f.writelines(converted_lines)
    
    # Copy and update other files
    for file in ["notes.json"]:
        src_file = os.path.join(source_dir, file)
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(output_dir, file))
    
    # Create dataset YAML
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    
    # Load class names
    notes_path = os.path.join(output_dir, "notes.json")
    if os.path.exists(notes_path):
        import json
        with open(notes_path, 'r') as f:
            notes = json.load(f)
        names = [category['name'] for category in notes.get('categories', [])]
    else:
        names = ['Parking Spot']
    
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n")
        f.write("names:\n")
        for i, name in enumerate(names):
            f.write(f"  {i}: {name}\n")
    
    print(f"Converted OBB dataset to regular YOLO format in {output_dir}")
    return output_dir

if __name__ == "__main__":
    convert_obb_to_regular_yolo()
