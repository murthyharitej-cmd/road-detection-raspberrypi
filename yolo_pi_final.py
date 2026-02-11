import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time

MODEL_PATH = "/home/haritej/road_proj/models/best.onnx"
IMG_SIZE = 320

print("Loading model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("Model loaded")

# Start Pi camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640,480)})
picam2.configure(config)
picam2.start()

print("Camera + Detection running")
print("Press Q to exit")

while True:

    frame = picam2.capture_array()

    # Convert RGBA -> BGR
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    orig = frame.copy()
    h, w = frame.shape[:2]

    # preprocess
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0
    img = np.transpose(img,(2,0,1))
    img = np.expand_dims(img,0)

    outputs = session.run(None,{input_name:img})
    preds = outputs[0][0]

    for det in preds:
        conf = det[4]
        if conf > 0.4:
            cx,cy,bw,bh = det[:4]

            x1 = int((cx-bw/2)*w)
            y1 = int((cy-bh/2)*h)
            x2 = int((cx+bw/2)*w)
            y2 = int((cy+bh/2)*h)

            cv2.rectangle(orig,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(orig,f"Pothole {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)

    cv2.imshow("AI POTHOLE DETECTION",orig)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()


