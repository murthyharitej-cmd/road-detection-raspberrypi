import cv2
import numpy as np
import onnxruntime as ort
import time

# --- CONFIG ---
MODEL_PATH = "../models/expert_fast.onnx"
VIDEO_PATH = "../videos/test.mp4"
OUTPUT_PATH = "../output/final_clean.mp4"
INPUT_SIZE = 320 

# --- LOAD ---
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(VIDEO_PATH)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

def postprocess(preds):
    boxes, confs = [], []
    # YOLOv8 output: [cx, cy, w, h, conf]
    for p in preds:
        score = p[4]
        if score > 0.40: # Higher threshold = less noise
            cx, cy, w, h = p[:4]
            # Convert to top-left format for OpenCV NMS
            x = int((cx - w/2) * (width / INPUT_SIZE))
            y = int((cy - h/2) * (height / INPUT_SIZE))
            w_px = int(w * (width / INPUT_SIZE))
            h_px = int(h * (height / INPUT_SIZE))
            
            # Road Filter: Only look at bottom 60%
            if y > (height * 0.4):
                boxes.append([x, y, w_px, h_px])
                confs.append(float(score))

    # --- THE MAGIC FIX: Non-Maximum Suppression ---
    # 0.45 means if boxes overlap by 45%, keep only the one with higher confidence
    indices = cv2.dnn.NMSBoxes(boxes, confs, 0.40, 0.45)
    
    final_results = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_results.append((boxes[i], confs[i]))
    return final_results

print("🚀 Running Clean & Fast Mode...")
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    # SPEED HACK: Only run AI on every 3rd frame
    # This triples your FPS immediately!
    if frame_count % 3 == 0:
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0 
        img = np.transpose(img, (2, 0, 1))
        
        outputs = session.run([output_name], {input_name: np.expand_dims(img, axis=0)})
        detections = postprocess(np.squeeze(outputs[0]).T)

        # Draw the clean boxes
        for (box, score) in detections:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, f"POTHOLE {score:.2f}", (x, y-10), 0, 0.6, (0, 0, 255), 2)

    out.write(frame)
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {frame_count} frames | Est. FPS: {frame_count/elapsed:.1f}", end="\r")

cap.release()
out.release()
print(f"\n✅ All done! Saved to {OUTPUT_PATH}")

