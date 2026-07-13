# batch_test_images.py
from ultralytics import YOLO
import cv2, os, sys
import numpy as np

MODEL_PATH = "best2.pt"
INPUT_DIR = "test_images"   # all test images go in this folder
OUT_DIR = "annotated_out"
IMG_SIZE = 640
CONF = 0.15
IOU = 0.45

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)
print("Class names:", model.names)

# list images
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]

if not files:
    print("No images found in test_images/. Add images and try again.")
    sys.exit(0)

for fname in files:
    path = os.path.join(INPUT_DIR, fname)
    print("\nTesting:", fname)

    img = cv2.imread(path)
    if img is None:
        print(" -> Could not read", fname)
        continue

    h, w = img.shape[:2]
    print(" Image resolution:", w, "x", h)

    results = model.predict(
        source=[img[:,:,::-1]],
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        verbose=False
    )
    r = results[0]

    out_img = img.copy()

    if r is None or r.boxes is None or len(r.boxes) == 0:
        print(" -> No detections")
    else:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        print(" -> Detections:", len(boxes))

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            score = float(scores[i])
            cls = int(classes[i])
            label = model.names.get(cls, str(cls))

            print(f"    {label} {score:.3f} ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

            cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            cv2.putText(out_img, f"{label} {score:.2f}", (int(x1), int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        save_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(save_path, out_img)
        print(" Saved annotated:", save_path)

print("\nDone. Check annotated_out/ folder.")
