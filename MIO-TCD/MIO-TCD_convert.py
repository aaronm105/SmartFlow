# this is the conversion script that 
# converts the MIO-TCD data into YOLO readable data 
# to be used to train YOLO models
import os
import pandas as pd
from PIL import Image

# specify path for the ground truth values
# specify path for the images and create a directory for labels
csv_path = "C:/Users/sebas/Desktop/MIO-TCD-Localization/gt_train.csv"
images_dir = "C:/Users/sebas/Desktop/MIO-TCD-Localization/test"
labels_dir = os.path.join(images_dir, "labels")

# make sure labels directory exists
os.makedirs(labels_dir, exist_ok=True)

# read the csv
df = pd.read_csv(csv_path, header=None)

# we are renaming columns from csv to be used for YOLO format
df.columns = ['image_id', 'class_name', 'xmin', 'ymin', 'xmax', 'ymax']

# mapping it out
unique_classes = sorted(df['class_name'].unique())
class_map = {name: idx for idx, name in enumerate(unique_classes)}
print("Class mapping:", class_map) # print the mapping before writing the .txt files

#for each img, obtain its coords from the csv table and convert/normalize them for YOLO

for img_id, group in df.groupby('image_id'):

    img_name = f"{int(img_id):08d}.jpg"
    img_path = os.path.join(images_dir, img_name)

    if not os.path.exists(img_path):
        print(f"Warning: missing {img_name}")
        continue

    img = Image.open(img_path)
    img_w, img_h = img.size

    yolo_lines = []
    for _, row in group.iterrows():
        x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        class_id = class_map[row['class_name']]

        # convert the coords to a value in b/w 0-1
        x_center = ((x_min + x_max) / 2) / img_w
        y_center = ((y_min + y_max) / 2) / img_h
        box_w = (x_max - x_min) / img_w
        box_h = (y_max - y_min) / img_h

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    # save the file and map it to the img
    label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))
