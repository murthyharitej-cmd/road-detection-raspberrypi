import cv2
import numpy as np
import onnxruntime as ort
import time
from picamera2 import Picamera2

# --- CONFIGURATION ---
MODEL_PATH = "../models/expert_fast.onnx"
INPUT_SIZE = 320 
CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45
FRAME_SKIP = 3  # Run AI every 3rd frame to boost speed
ALPHA = 0.1     # FPS smoothing factor (lower = more stable)

# --- INITIALIZE AI ---
# Using 4 threads to maximize the Raspberry Pi's Quad-core CPU
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4 
try:
    sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print("✅ AI Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- INITIALIZE CAMERA ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)})
picam2.configure(config)
picam2.start()

# --- HELPER FUNCTIONS ---
def postprocess(preds, w, h):
    boxes, confs = [], []
    for p in preds:
        if p[4] > CONF_THRESHOLD:
            cx, cy, bw, bh = p[:4]
            # Rescale coordinates to original frame size
            x = int((cx - bw/2) * (w / INPUT_SIZE))
            y = int((cy - bh/2) * (h / INPUT_SIZE))
            # Road Filter: Only detect in the bottom 60% of the screen
            if y > (h * 0.40): 
                boxes.append([x, y, int(bw * (w / INPUT_SIZE)), int(bh * (h / INPUT_SIZE))])
                confs.append(float(p[4]))
    
    indices = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESHOLD, NMS_THRESHOLD)
    return [(boxes[i], confs[i]) for i in (indices.flatten() if len(indices) > 0 else [])]

# --- MAIN LOOP ---
fps_avg = 0
frame_count = 0
current_dets = []
prev_time = time.time()

print("🚀 Real-Time Pothole Detection Active (VNC Mode)")
print("Press 'q' in the video window to quit.")

try:
    while True:
        # 1. Capture and convert frame
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]
        frame_count += 1

        # 2. Stable FPS Calculation (Moving Average)
        new_time = time.time()
        instant_fps = 1 / (new_time - prev_time)
        prev_time = new_time
        fps_avg = (ALPHA * instant_fps) + ((1 - ALPHA) * fps_avg)

        # 3. AI Inference (Every X frames)
        if frame_count % FRAME_SKIP == 0:
            # Pre-process
            blob = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
            blob = blob.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
            
            # Run Inference
            preds = np.squeeze(sess.run([out_name], {in_name: blob})[0]).T
            current_dets = postprocess(preds, w, h)

        # 4. Draw Detections
        for (box, score) in current_dets:
            x, y, bw, bh = box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Pothole {score:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5. Overlay FPS Counter
        cv2.rectangle(frame, (10, 10), (160, 50), (0, 0, 0), -1) # Black background for readability
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 6. Display Output
        cv2.imshow("Pothole Detection Live", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    print("Cleaning up...")
    picam2.stop()
    cv2.destroyAllWindows()
