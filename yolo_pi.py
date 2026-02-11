import cv2
import numpy as np
import onnxruntime as ort

model_path = "models/best.onnx"
video_path = "videos/pothole.mp4"
output_path = "output/result.mp4"

print("Loading model...")

# prevent CPU overload freeze
so = ort.SessionOptions()
so.intra_op_num_threads = 2
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(video_path)

# get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
video_fps = cap.get(cv2.CAP_PROP_FPS)

if video_fps <= 0:
    video_fps = 30

print(f"Input video FPS: {video_fps:.1f}")

# target processing fps
target_fps = 7   # 🔥 you can change 5–8 range
frame_skip = int(video_fps / target_fps)

if frame_skip < 1:
    frame_skip = 1

print(f"Detection every {frame_skip} frames (auto-adjusted)")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_width, frame_height))

imgsz = 320
frame_count = 0

last_boxes = []
last_conf = []

print("Adaptive smooth detection running...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    orig = frame.copy()
    h, w = frame.shape[:2]

    # adaptive detection interval
    if frame_count % frame_skip == 0:

        img = cv2.resize(frame, (imgsz, imgsz))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
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

            if conf > 0.4:
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

                last_boxes.append([x,y,w_box,h_box])
                last_conf.append(confidences[i])

    # draw detections smoothly every frame
    for i in range(len(last_boxes)):
        x,y,w_box,h_box = last_boxes[i]
        cv2.rectangle(orig,(x,y),(x+w_box,y+h_box),(0,255,0),2)
        cv2.putText(orig,f"Pothole {last_conf[i]:.2f}",
                    (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("AI Pothole Detection - Raspberry Pi", orig)
    out.write(orig)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Saved video in output/result.mp4")

