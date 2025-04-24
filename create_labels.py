import os
import glob
from pathlib import Path
import shutil
import yaml

def create_yolo_dataset_structure(base_dir):
    """
    Create the standard YOLO directory structure.
    
    Parameters:
        base_dir (str): Base directory for the dataset
    """
    # Create main directories
    os.makedirs(os.path.join(base_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "val"), exist_ok=True)
    
    print(f"YOLO directory structure created at: {base_dir}")

def create_sample_labels(image_dir, label_dir, class_id=0):
    """
    Create sample label files for existing images.
    
    Parameters:
        image_dir (str): Directory of images
        label_dir (str): Directory for labels
        class_id (int): Class ID (0 for soccer player)
    """
    # Ensure the label directory exists
    os.makedirs(label_dir, exist_ok=True)
    
    # Get a list of all images
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_files:
        print(f"Warning: No images found in directory {image_dir}")
        return
    
    # Create a label file for each image
    for img_path in image_files:
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)
        
        # Create a default label (rectangle in the center of the image)
        # YOLO format: <class_id> <x_center> <y_center> <width> <height>
        # All values are normalized between 0 and 1
        with open(label_path, "w") as f:
            # Player approximately in the center of the image
            f.write(f"{class_id} 0.5 0.5 0.1 0.2\n")
            # Additional players can be added
            f.write(f"{class_id} 0.3 0.6 0.1 0.2\n")
            f.write(f"{class_id} 0.7 0.4 0.1 0.2\n")
    
    print(f"Created {len(image_files)} label files in {label_dir}")

def create_data_yaml(base_dir, class_names=None):
    """
    Create a data.yaml file to configure the dataset.
    
    Parameters:
        base_dir (str): Base directory for the dataset
        class_names (list): List of class names
    """
    if class_names is None:
        class_names = ["player"]  # Default class is "player"
    
    # Create content for the data.yaml file
    data_yaml = {
        "path": os.path.abspath(base_dir),  # Absolute path to the base directory
        "train": "images/train",  # Path to the training images folder
        "val": "images/val",      # Path to the validation images folder
        "names": {i: name for i, name in enumerate(class_names)}  # Class names
    }
    
    # Save the data.yaml file
    yaml_path = os.path.join(base_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print(f"Created data.yaml file at: {yaml_path}")
    return yaml_path

def copy_images_if_empty(base_dir, sample_image_path):
    """
    Copy a sample image to the image folders if they are empty.
    
    Parameters:
        base_dir (str): Base directory for the dataset
        sample_image_path (str): Path to the sample image
    """
    if not os.path.exists(sample_image_path):
        print(f"Warning: Sample image not found: {sample_image_path}")
        return
    
    # Check image folders and add sample images if they are empty
    train_dir = os.path.join(base_dir, "images", "train")
    val_dir = os.path.join(base_dir, "images", "val")
    
    if not glob.glob(os.path.join(train_dir, "*")):
        # Copy sample image to the training folder
        for i in range(5):  # Create 5 training images
            dest_path = os.path.join(train_dir, f"train_{i}.jpg")
            shutil.copy(sample_image_path, dest_path)
        print(f"Copied 5 sample images to training folder")
    
    if not glob.glob(os.path.join(val_dir, "*")):
        # Copy sample image to the validation folder
        for i in range(3):  # Create 3 validation images
            dest_path = os.path.join(val_dir, f"val_{i}.jpg")
            shutil.copy(sample_image_path, dest_path)
        print(f"Copied 3 sample images to validation folder")

def fix_missing_labels(dataset_dir, sample_image_path=None):
    """
    Fix missing labels in a YOLO dataset.
    
    Parameters:
        dataset_dir (str): Dataset directory
        sample_image_path (str): Path to a sample image (optional)
    
    Returns:
        str: Path to the data.yaml file
    """
    # Create the folder structure if it doesn't exist
    create_yolo_dataset_structure(dataset_dir)
    
    # If a sample image is provided, copy it to empty folders
    if sample_image_path:
        copy_images_if_empty(dataset_dir, sample_image_path)
    
    # Create label files for existing images
    train_img_dir = os.path.join(dataset_dir, "images", "train")
    train_label_dir = os.path.join(dataset_dir, "labels", "train")
    val_img_dir = os.path.join(dataset_dir, "images", "val")
    val_label_dir = os.path.join(dataset_dir, "labels", "val")
    
    create_sample_labels(train_img_dir, train_label_dir)
    create_sample_labels(val_img_dir, val_label_dir)
    
    # Create data.yaml file
    class_names = ["player", "ball", "referee"]  # This list can be modified as needed
    yaml_path = create_data_yaml(dataset_dir, class_names)
    
    print("\nSuccessfully fixed missing labels!")
    print(f"You can now use the data.yaml file at: {yaml_path}")
    print("To evaluate the model using the command:")
    print(f"python evaluate_model.py [model_path] {yaml_path}")
    
    return yaml_path

if __name__ == "__main__":
    import sys
    
    # Default dataset directory
    default_dataset_dir = "yolo_dataset"
    
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    else:
        dataset_dir = default_dataset_dir
    
    # Path to a sample image (optional)
    sample_image_path = None
    if len(sys.argv) > 2:
        sample_image_path = sys.argv[2]
    
    # Fix missing labels
    fix_missing_labels(dataset_dir, sample_image_path)