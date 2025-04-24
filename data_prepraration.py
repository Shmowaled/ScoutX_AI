from datasets import load_dataset
import os
import shutil
from pathlib import Path
import random

# تحديد المسارات
base_dir = Path("Voxel51/Football-Player-Segmentation")
images_dir = base_dir / "images" 
labels_dir = base_dir / "labels"

# التأكد من وجود مجلدات للتدريب والتقييم
train_images_dir = images_dir / "train"
val_images_dir = images_dir / "val"
train_labels_dir = labels_dir / "train"
val_labels_dir = labels_dir / "val"

# إنشاء مجلدات التقييم إذا لم تكن موجودة
val_images_dir.mkdir(parents=True, exist_ok=True)
val_labels_dir.mkdir(parents=True, exist_ok=True)

# الحصول على جميع الصور والملفات الخاصة بالتدريب
image_files = list(train_images_dir.glob("*.jpg"))  # أو أي صيغة أخرى مثل png
label_files = list(train_labels_dir.glob("*.txt"))

# تحديد عدد الصور التي سيتم تخصيصها للتقييم (مثلاً 20%)
num_val = int(0.2 * len(image_files))  # 20% للتقييم

# اختيار عشوائي للصور التي ستكون للتقييم
val_images = random.sample(image_files, num_val)

# نقل الصور والملفات الخاصة بها إلى مجلد التقييم
for img in val_images:
    shutil.move(img, val_images_dir / img.name)
    label_file = train_labels_dir / img.stem + ".txt"
    shutil.move(label_file, val_labels_dir / label_file.name)
