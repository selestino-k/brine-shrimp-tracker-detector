import cv2 as cv
import os
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from glob import glob
import yaml
from ultralytics import YOLO

def create_dataset_structure(base_dir):
    """Create the directory structure required for YOLO."""
    dirs = ['images/train', 'images/val', 'images/test', 
            'labels/train', 'labels/val', 'labels/test']
    
    for dir_path in dirs:
        os.makedirs(os.path.join(base_dir, dir_path), exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")


def process_dataset(source_dir, output_dir, class_names):
    """Process the entire dataset and prepare it for YOLO training."""
    # Create dataset structure
    create_dataset_structure(output_dir)
    
    # Get all image files
    image_files = glob(os.path.join(source_dir, 'images', '*.jpg')) + \
                 glob(os.path.join(source_dir, 'images', '*.png')) + \
                 glob(os.path.join(source_dir, 'images', '*.jpeg'))
    
    # Split dataset
    train_files,  = train_test_split(image_files, test_size=0.3, random_state=42)
    val_files,  = train_test_split(image_files, test_size=0.5, random_state=42)
    
    num_classes = len(class_names)
    
    # Process train set
    print("Processing training set...")
    process_file_set(train_files, source_dir, output_dir, 'train', num_classes)
    
    # Process validation set
    print("Processing validation set...")
    process_file_set(val_files, source_dir, output_dir, 'val', num_classes)
    
    
    
    # Create dataset.yaml file
    create_yaml_config(output_dir, class_names)
    
    print("Dataset preprocessing complete!")

def process_file_set(files, source_dir, output_dir, set_type, num_classes):
    """Process a set of files (train, val, or test)."""
    for img_path in files:
        # Get file basename
        base_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Copy image
        dest_img_path = os.path.join(output_dir, 'images', set_type, base_name)
        shutil.copy(img_path, dest_img_path)
        
        # Process annotation
        ann_path = os.path.join(source_dir, 'labels', f"{name_without_ext}.txt")
        
        if os.path.exists(ann_path):
            # Read image to get dimensions
            img = cv.imread(img_path)
            img_height, img_width = img.shape[:2]
            
            # Convert annotation to YOLO format
            yolo_annotations = convert_to_yolo_format(ann_path, img_width, img_height, num_classes)
            
            # Save YOLO format annotation
            dest_ann_path = os.path.join(output_dir, 'labels', set_type, f"{name_without_ext}.txt")
            with open(dest_ann_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        else:
            print(f"Warning: No annotation found for {img_path}")

def create_yaml_config(output_dir, class_names):
    """Create YAML configuration file for dataset."""
    config = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"Created dataset.yaml in {output_dir}")

def validate_dataset(dataset_dir):
    """Validate that the dataset is properly formatted."""
    yaml_file = os.path.join(dataset_dir, 'dataset.yaml')
    
    if not os.path.exists(yaml_file):
        print(f"Error: dataset.yaml not found in {dataset_dir}")
        return False
    
    # Load YAML config
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required keys
    required_keys = ['path', 'train', 'val', 'names']
    for key in required_keys:
        if key not in config:
            print(f"Error: '{key}' not found in dataset.yaml")
            return False
    
    # Check image and label directories
    sets = ['train', 'val']
    for set_type in sets:
        img_dir = os.path.join(dataset_dir, 'images', set_type)
        label_dir = os.path.join(dataset_dir, 'labels', set_type)
        
        if not os.path.exists(img_dir):
            print(f"Error: {img_dir} not found")
            return False
        
        if not os.path.exists(label_dir):
            print(f"Error: {label_dir} not found")
            return False
        
        # Check if images have corresponding labels
        img_files = glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.png'))
        for img_file in img_files:
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            label_file = os.path.join(label_dir, f"{base_name}.txt")
            
            if not os.path.exists(label_file):
                print(f"Warning: No label file for {img_file}")
    
    print("Dataset validation completed successfully.")
    return True

if __name__ == "__main__":
    # Define your class names
    class_names = ['Brine Shrimp',]
    
    # Define source and output directories
    source_dir = '/small-shrimp-data'  # Change this to your source dataset directory
    output_dir = '/data'  # Change this to your desired output directory
    
    # Process the dataset
    process_dataset(source_dir, output_dir, class_names)
    
    # Validate the dataset
    validate_dataset(output_dir)
    
    print("Dataset is ready for training with Ultralytics YOLOv11!")

