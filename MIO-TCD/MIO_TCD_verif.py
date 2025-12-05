# this short script verifies that MIO-TCD localization dataset was converted correctly
# after conversion, we can open an image and map its YOLO converted class ID and bounding box coords
# then display it using opencv
import cv2

img_path = "C:/Users/sebas/Desktop/MIO-TCD-Localization/train/00000010.jpg"
label_path = "C:/Users/sebas/Desktop/MIO-TCD-Localization/train/labels/00000010.txt"

img = cv2.imread(img_path)
h, w = img.shape[:2]

with open(label_path) as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.strip().split())
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # make all boxes green
        cv2.putText(img, str(int(cls)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("label testing", img)
cv2.waitKey(0)
cv2.destroyAllWindows()