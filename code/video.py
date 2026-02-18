import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# --- SETUP ---
MODEL = "../models/expert_fast.onnx"
SIZE = 320 
vid_name = input("Video name: ")
path = f"../videos/{vid_name}"

if not os.path.exists(path): exit("File not found")

# Optimization: Use all 4 CPU cores
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
sess = ort.InferenceSession(MODEL, sess_options=opts, providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

cap = cv2.VideoCapture(path)
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(f"../output/out_{vid_name}", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))

print(f"⚡ Running {vid_name} at 320px...")
count, start, dets = 0, time.time(), []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    count += 1

    # Only run AI every 3rd frame to hit 7+ FPS
    if count % 3 == 0:
        # Fast resize + normalize
        blob = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        
        preds = np.squeeze(sess.run([out_name], {in_name: blob})[0]).T
        
        # Filter boxes
        boxes, confs = [], []
        for p in preds:
            if p[4] > 0.35:
                cx, cy, bw, bh = p[:4]
                x, y = int((cx-bw/2)*(w/SIZE)), int((cy-bh/2)*(h/SIZE))
                if y > (h * 0.45): # Road filter
                    boxes.append([x, y, int(bw*(w/SIZE)), int(bh*(h/SIZE))])
                    confs.append(float(p[4]))
        
        idx = cv2.dnn.NMSBoxes(boxes, confs, 0.35, 0.45)
        dets = [(boxes[i], confs[i]) for i in (idx.flatten() if len(idx) > 0 else [])]

    # Draw detections
    for (box, score) in dets:
        bx, by, bw, bh = box
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
        cv2.putText(frame, f"P {score:.2f}", (bx, by-5), 0, 0.4, (0, 255, 0), 1)

    out.write(frame)
    if count % 15 == 0:
        fps = count / (time.time() - start)
        print(f"Frame: {count} | FPS: {fps:.1f}", end="\r")

cap.release()
out.release()
print(f"\n✅ Finished. Output in ../output/out_{vid_name}")
