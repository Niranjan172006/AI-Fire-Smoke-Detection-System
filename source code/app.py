# app.py - YOLOv8 Fire/Smoke Detection ML Server

from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import traceback
import torch

app = Flask(__name__)

# --------------- MODEL CONFIG -----------------
MODEL_PATH = "best2.pt"  # your trained YOLOv8 model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
MAX_DETS = 100
# ----------------------------------------------

print("[ml_server] Loading YOLOv8 model:", MODEL_PATH)
model = YOLO(MODEL_PATH)
model.to(DEVICE)


# ----------- YOLO INFERENCE FUNCTION -----------
def run_detection(image_bgr):
    try:
        H, W = image_bgr.shape[:2]

        # Convert BGR → RGB
        image_rgb = image_bgr[:, :, ::-1]

        results = model.predict(
            source=image_rgb,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            max_det=MAX_DETS,
            device=DEVICE,
            verbose=False
        )

        detections = []
        r = results[0]

        if r is None or r.boxes is None:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        labels = model.names

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            detections.append({
                "x": float(x1 / W),
                "y": float(y1 / H),
                "w": float(w / W),
                "h": float(h / H),
                "score": float(scores[i]),
                "label": labels[int(classes[i])]
            })

        return detections

    except Exception as e:
        print("Detection error:", e)
        traceback.print_exc()
        return []
# ------------------------------------------------


# ------------- API: REALTIME DETECTION ----------
@app.route("/realtime-detect", methods=["POST"])
def realtime_detect():
    try:
        data = request.json
        frame = data["frame"]

        # Strip "data:image/jpeg;base64,"
        _, b64data = frame.split(",", 1)

        frame_bytes = base64.b64decode(b64data)
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        img_np = np.array(img)[:, :, ::-1]  # RGB → BGR

        detections = run_detection(img_np)

        return jsonify({"detections": detections})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# ------------------------------------------------


# ------------- API: IMAGE DETECTION --------------
@app.route("/detect-image", methods=["POST"])
def detect_image():
    try:
        data = request.json
        b64 = data["image_base64"]

        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)[:, :, ::-1]

        detections = run_detection(img_np)

        return jsonify({"detections": detections})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# -------------------------------------------------


# ------------- API: VIDEO DETECTION --------------
@app.route("/detect-video", methods=["POST"])
def detect_video():
    try:
        data = request.json
        video_b64 = data["video_base64"]

        video_bytes = base64.b64decode(video_b64)
        temp_path = "uploaded_video.mp4"

        with open(temp_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_path)
        all_results = []
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process 1 frame every 5 frames (speed-up)
            if frame_id % 5 == 0:
                det = run_detection(frame)
                if det:
                    all_results.append({"frame": frame_id, "detections": det})

            frame_id += 1

        cap.release()
        return jsonify({"video_detections": all_results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# --------------------------------------------------


# ------------- START SERVER -----------------------
if __name__ == "__main__":
    print("[ml_server] Server started on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
# --------------------------------------------------
