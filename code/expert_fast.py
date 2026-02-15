import cv2
import numpy as np
import onnxruntime as ort
import time

# --- CONFIG ---
MODEL_PATH = "../models/expert_fast.onnx"
VIDEO_PATH = "../videos/test.mp4"
OUTPUT_PATH = "../output/fast_result.mp4"
INPUT_SIZE = 320  # Fast resolution

# --- LOAD ---
# Using 'internal' optimization levels for ONNX
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(VIDEO_PATH)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

print("🚀 Running in High-Speed Expert Mode...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # FAST PREPROCESS
    # We use cv2.INTER_NEAREST because it's the fastest resizing method
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0 
    img = np.transpose(img, (2, 0, 1))
    input_tensor = np.expand_dims(img, axis=0)

    # INFERENCE
    outputs = session.run([output_name], {input_name: input_tensor})
    preds = np.squeeze(outputs[0]).T

    # FILTER & DRAW
    # We only look at the bottom 60% of the road to save time
    for p in preds:
        if p[4] > 0.30: # Confidence threshold
            cx, cy, w, h = p[:4]
            # Map coordinates
            x = int((cx - w/2) * (width / INPUT_SIZE))
            y = int((cy - h/2) * (height / INPUT_SIZE))
            
            if y > (height * 0.4): # Road Filter
                cv2.rectangle(frame, (x, y), (x + int(w*(width/INPUT_SIZE)), y + int(h*(height/INPUT_SIZE))), (0, 0, 255), 3)

    out.write(frame)

cap.release()
out.release()
print(f"✅ Finished! Output: {OUTPUT_PATH}")
