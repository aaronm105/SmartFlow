# this script splits the 100K+ photos into 10 percent val and 90 percent train for YOLO
import os
import shutil
import random

train_dir = "C:/Users/sebas/Desktop/MIO-TCD-Localization/train"
val_dir = "C:/Users/sebas/Desktop/MIO-TCD-Localization/val"
labels_dir = os.path.join(train_dir, "labels")
val_labels_dir = os.path.join(val_dir, "labels")

val_split = 0.10  # 10% of images for validation

os.makedirs(val_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

all_images = [f for f in os.listdir(train_dir) if f.endswith(".jpg")]

# randomly select images for 10 percent
val_count = int(len(all_images) * val_split)
val_images = random.sample(all_images, val_count)
print(f"Selected {val_count} images for validation.")

for img in val_images:
    # move the img
    shutil.move(os.path.join(train_dir, img), os.path.join(val_dir, img))
    
    # move its label
    label_file = img.replace(".jpg", ".txt")
    label_src = os.path.join(labels_dir, label_file)
    label_dst = os.path.join(val_labels_dir, label_file)
    
    if os.path.exists(label_src):
        shutil.move(label_src, label_dst)
    else:
        print(f"Warning: Label file missing for {img}")

print("Train/Val split complete!")
print(f"Train images remaining: {len(os.listdir(train_dir))}")
print(f"Validation images: {len(os.listdir(val_dir))}")
