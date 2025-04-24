from datasets import load_dataset
import os
from PIL import Image

from ultralytics import YOLO
model = YOLO("yolov8x.pt")  # Best accuracy, pre-trained
results = model.predict(source="yolo_dataset/images", save=True, save_txt=True, conf=0.5)

# Load the dataset (this automatically downloads it)
dataset = load_dataset("Voxel51/Football-Player-Segmentation", split="train")

# Create folders
os.makedirs("yolo_dataset/images", exist_ok=True)
os.makedirs("yolo_dataset/masks", exist_ok=True)

# Save images and masks
for i, sample in enumerate(dataset):
    img = sample["image"]
    mask = sample["segmentation_mask"]

    img.save(f"yolo_dataset/images/image_{i}.jpg")
    mask.save(f"yolo_dataset/masks/mask_{i}.png")

    if i > 300:  # Limit for quick testing
        break

print("Images and masks saved!")
