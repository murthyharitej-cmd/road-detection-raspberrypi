import cv2
import numpy as np
import onnxruntime as ort
import time

# --- CONFIG ---
MODEL_PATH = "../models/expert_pothole.onnx"
VIDEO_PATH = "../videos/test.mp4"
OUTPUT_PATH = "../output/expert_final_result.mp4"
INPUT_SIZE = 640

# --- LOAD ---
print("Initializing ONNX Runtime...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(VIDEO_PATH)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

def postprocess(preds, frame):
    boxes, confs = [], []
    # Standard YOLOv8 output: [cx, cy, w, h, class_conf...]
    # We filter with a higher confidence to stop the "junk" boxes
    for p in preds:
        conf = p[4]
        if conf > 0.35: 
            cx, cy, w, h = p[:4]
            x = int((cx - w/2) * (width / INPUT_SIZE))
            y = int((cy - h/2) * (height / INPUT_SIZE))
            
            # ROAD FILTER: Ignore detections in the top 45% (sky/horizon)
            if y > (height * 0.45):
                boxes.append([x, y, int(w * (width / INPUT_SIZE)), int(h * (height / INPUT_SIZE))])
                confs.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, confs, 0.35, 0.45)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            # Red Box for Potholes
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, f"POTHOLE {confs[i]:.2f}", (x, y-10), 0, 0.7, (0, 0, 255), 2)
    return frame

print("🚀 Starting Inference with Expert Model...")
start_time = time.time()
frame_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        # Preprocess
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0 
        img = np.transpose(img, (2, 0, 1))
        input_tensor = np.expand_dims(img, axis=0)

        # Inference
        outputs = session.run([output_name], {input_name: input_tensor})
        preds = np.squeeze(outputs[0]).T

        # Draw & Write
        frame = postprocess(preds, frame)
        out.write(frame)
        
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...", end="\r")

except KeyboardInterrupt:
    print("\nStopped by user.")

cap.release()
out.release()
print(f"\n✅ Finished! Video saved to {OUTPUT_PATH}")
