import cv2
import numpy as np
import onnxruntime as ort
import time

MODEL_PATH = "models/best.onnx"
IMG_SIZE = 320

print("Loading model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("Model loaded")

# Picamera via libcamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera failed")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Detection started")
print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # FIX: convert 4 channel to 3 channel
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    orig = frame.copy()

    # resize for model
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    outputs = session.run(None, {input_name: img})
    preds = outputs[0][0]

    for det in preds:
        conf = det[4]
        if conf > 0.4:
            x1,y1,x2,y2 = map(int, det[:4])

            # scale back to original frame
            x1 = int(x1 * orig.shape[1] / IMG_SIZE)
            x2 = int(x2 * orig.shape[1] / IMG_SIZE)
            y1 = int(y1 * orig.shape[0] / IMG_SIZE)
            y2 = int(y2 * orig.shape[0] / IMG_SIZE)

            cv2.rectangle(orig,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(orig,f"Pothole {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)

    cv2.imshow("AI POTHOLE DETECTION", orig)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

