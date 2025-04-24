import os, shutil
from sklearn.model_selection import train_test_split

images = os.listdir("yolo_dataset/images")
train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
    os.makedirs(f"yolo_dataset/images/{split}", exist_ok=True)
    os.makedirs(f"yolo_dataset/labels/{split}", exist_ok=True)
    for img in img_list:
        base = os.path.splitext(img)[0]
        shutil.move(f"yolo_dataset/images/{img}", f"yolo_dataset/images/{split}/{img}")
        shutil.move(f"yolo_dataset/labels/{base}.txt", f"yolo_dataset/labels/{split}/{base}.txt")
