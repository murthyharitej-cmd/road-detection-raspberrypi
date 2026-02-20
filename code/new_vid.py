import cv2
import numpy as np
import onnxruntime as ort
import time
import os

MODEL = "../models/model_a.onnx" 
SIZE = 480  
vid_name = input("Video name: ")
path = f"../videos/{vid_name}"

if not os.path.exists(path): exit("File not found")

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
sess = ort.InferenceSession(MODEL, sess_options=opts, providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

cap = cv2.VideoCapture(path)
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(f"../output/out_{vid_name}", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))

HAZARD_MAP = {
    6: "POTHOLE",
    16: "POTHOLE",
    2: "OBSTACLE",
    0: "ANIMAL"
}

print(f"⚡ Detecting Road Hazards in {vid_name}...")
count, start, dets = 0, time.time(), []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    count += 1

    if count % 3 == 0:
        blob = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        
        outputs = sess.run([out_name], {in_name: blob})[0]
        preds = np.squeeze(outputs).T 
        
        boxes, confs, labels = [], [], []
        for p in preds:
            class_probs = p[4:]
            best_class_idx = np.argmax(class_probs)
            score = class_probs[best_class_idx]
            
            if score > 0.40 and best_class_idx in HAZARD_MAP:
                cx, cy, bw, bh = p[:4]
                x = int((cx - bw/2) * (w / SIZE))
                y = int((cy - bh/2) * (h / SIZE))
                
                if y > (h * 0.40): 
                    boxes.append([x, y, int(bw * (w / SIZE)), int(bh * (h / SIZE))])
                    confs.append(float(score))
                    labels.append(HAZARD_MAP[best_class_idx])
        
        idx = cv2.dnn.NMSBoxes(boxes, confs, 0.40, 0.45)
        dets = [(boxes[i], confs[i], labels[i]) for i in (idx.flatten() if len(idx) > 0 else [])]

    for (box, score, label) in dets:
        bx, by, bw, bh = box
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
        cv2.putText(frame, f"{label} {score:.2f}", (bx, by-10), 0, 0.6, (0, 0, 255), 2)

    out.write(frame)
    if count % 15 == 0:
        fps = count / (time.time() - start)
        print(f"Frame: {count} | FPS: {fps:.1f}", end="\r")

cap.release()
out.release()
print(f"\n✅ Hazard Detection Complete. Saved as out_{vid_name}")
