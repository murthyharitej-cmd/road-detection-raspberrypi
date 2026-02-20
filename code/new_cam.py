import cv2
import numpy as np
import onnxruntime as ort
import time
from picamera2 import Picamera2

MODEL_PATH = "../models/model_a.onnx"
INPUT_SIZE = 480    
CONF_THRESHOLD = 0.40
NMS_THRESHOLD = 0.45
FRAME_SKIP = 3      
ALPHA = 0.05       

HAZARD_MAP = {
    6: "POTHOLE",
    16: "POTHOLE", 
    2: "OBSTACLE",
    0: "ANIMAL"
}

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4 
try:
    sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print("✅ Road Hazard Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)})
picam2.configure(config)
picam2.start()

def postprocess(preds, w, h):
    boxes, confs, labels = [], [], []
    for p in preds:
        class_probs = p[4:]
        best_class_idx = np.argmax(class_probs)
        score = class_probs[best_class_idx]
        
        if score > CONF_THRESHOLD and best_class_idx in HAZARD_MAP:
            cx, cy, bw, bh = p[:4]
            x = int((cx - bw/2) * (w / INPUT_SIZE))
            y = int((cy - bh/2) * (h / INPUT_SIZE))
            
            if y > (h * 0.40): 
                boxes.append([x, y, int(bw * (w / INPUT_SIZE)), int(bh * (h / INPUT_SIZE))])
                confs.append(float(score))
                labels.append(HAZARD_MAP[best_class_idx])
    
    indices = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESHOLD, NMS_THRESHOLD)
    return [(boxes[i], confs[i], labels[i]) for i in (indices.flatten() if len(indices) > 0 else [])]

fps_avg = 0
frame_count = 0
current_dets = []
prev_time = time.time()

print("🚀 Real-Time Hazard Detection Active")
print("Targeting: Potholes, Obstacles, and Animals")

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]
        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            blob = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            blob = blob.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
            
            outputs = sess.run([out_name], {in_name: blob})[0]
            preds = np.squeeze(outputs).T 
            current_dets = postprocess(preds, w, h)

        for (box, score, label) in current_dets:
            bx, by, bw, bh = box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (bx, by - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        new_time = time.time()
        instant_fps = 1 / (new_time - prev_time)
        prev_time = new_time
        fps_avg = (ALPHA * instant_fps) + ((1 - ALPHA) * fps_avg)
        
        cv2.rectangle(frame, (10, 10), (160, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Road Hazard Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
