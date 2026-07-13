# run_inference_debug.py
from ultralytics import YOLO
import cv2
import numpy as np
import os

MODEL_PATH = "best2.pt"
IMG_PATH = "test.jpg"
OUT_PATH = "annotated_test.jpg"
IMG_SIZE = 640
CONF = 0.25
IOU = 0.45

print("Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

print("Model class names:", getattr(model, "names", {}))

# read with cv2 (BGR)
img = cv2.imread(IMG_PATH)
if img is None:
    raise SystemExit(f"Could not read image file: {IMG_PATH}")

H, W = img.shape[:2]
print("Image size:", W, "x", H)

# run prediction
results = model.predict(source=[img[:,:,::-1]], imgsz=IMG_SIZE, conf=CONF, iou=IOU, verbose=False)
r = results[0]

# If no detections, r.boxes may be empty
if r is None or r.boxes is None or len(r.boxes) == 0:
    print("No boxes detected by model.")
else:
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    names = model.names
    print("Detections count:", len(boxes))
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        sc = float(scores[i])
        cl = int(classes[i])
        label = names.get(cl, str(cl))
        print(f"Box {i}: {label} score={sc:.3f} xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
        # draw box on image (BGR)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.putText(img, f"{label} {sc:.2f}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# save annotated result
cv2.imwrite(OUT_PATH, img)
print("Annotated image saved to:", OUT_PATH)
