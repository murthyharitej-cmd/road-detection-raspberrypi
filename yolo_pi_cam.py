import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import os
from datetime import datetime
import time

# ------------------ SETTINGS ------------------
model_path = "/home/haritej/road_proj/models/best_int8.onnx"
imgsz = 320
confidence_threshold = 0.4
inference_interval = 0.35   # seconds between detections (~6–8 FPS detection)
# ------------------------------------------------

print("Loading optimized ONNX model...")

# ONNX runtime optimization
so = ort.SessionOptions()
so.intra_op_num_threads = 3
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    model_path,
    sess_options=so,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# start Pi camera
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
)
picam2.start()

# session output folder
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_folder = f"output/session_{timestamp}"
os.makedirs(session_folder, exist_ok=True)

print(f"Saving detections → {session_folder}")

last_boxes = []
last_conf = []
last_inference_time = 0
save_count = 0

print("Optimized LIVE pothole detection started...")
print("Press ESC to exit")

# ------------------ MAIN LOOP ------------------
while True:

    frame = picam2.capture_array()
    orig = frame.copy()
    h, w = frame.shape[:2]

    current_time = time.time()

    # run inference at controlled interval
    if current_time - last_inference_time > inference_interval:

        last_inference_time = current_time

        img = cv2.resize(frame, (imgsz, imgsz))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = session.run(None, {input_name: img})
        preds = outputs[0]
        preds = np.squeeze(preds).T

        boxes = []
        confidences = []

        for row in preds:
            x, y, bw, bh = row[:4]
            scores = row[4:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if conf > confidence_threshold:

                x1 = int((x - bw/2) * w / imgsz)
                y1 = int((y - bh/2) * h / imgsz)
                x2 = int((x + bw/2) * w / imgsz)
                y2 = int((y + bh/2) * h / imgsz)

                boxes.append([x1, y1, x2-x1, y2-y1])
                confidences.append(float(conf))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

        last_boxes = []
        last_conf = []

        if len(indices) > 0:
            for i in indices:
                i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                x, y, w_box, h_box = boxes[i]

                last_boxes.append([x, y, w_box, h_box])
                last_conf.append(confidences[i])

            # save detected frame
            save_count += 1
            save_path = f"{session_folder}/pothole_{save_count:03d}.jpg"
            cv2.imwrite(save_path, orig)

    # draw boxes smoothly every frame
    for i in range(len(last_boxes)):
        x, y, w_box, h_box = last_boxes[i]
        cv2.rectangle(orig, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
        cv2.putText(orig, f"Pothole {last_conf[i]:.2f}",
                    (x, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    cv2.imshow("AI POTHOLE DETECTION — OPTIMIZED", orig)

    if cv2.waitKey(1) == 27:
        break

# cleanup
picam2.stop()
cv2.destroyAllWindows()

print("Session finished")
print(f"Saved pothole frames: {save_count}")

