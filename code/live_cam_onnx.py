from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import time
import os
from datetime import datetime

# Load INT8 ONNX model
model = YOLO("../models/best_int8.onnx")

# Create output folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"../outputs/live_{timestamp}"
os.makedirs(output_folder, exist_ok=True)

print("✅ Using model: best_int8.onnx")
print("✅ Output folder:", output_folder)

# Start Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

time.sleep(2)

frame_count = 0
saved_count = 0
conf_threshold = 0.5

print("🚀 Live hazard detection started. Press Q to exit.")

while True:
    frame = picam2.capture_array()
    frame_count += 1

    # YOLO inference
    results = model(frame, conf=conf_threshold)

    # If pothole/obstacle detected → save frame
    if len(results[0].boxes) > 0:
        saved_count += 1
        annotated = results[0].plot()

        cv2.imwrite(
            f"{output_folder}/frame_{frame_count}.jpg",
            annotated
        )

        print(f"⚠️ Hazard detected → saved frame {frame_count}")

    # Show live OpenCV window with bounding boxes
    cv2.imshow("Road Hazard Detection (ONNX)", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()

print("\n✅ Finished")
print("Frames processed:", frame_count)
print("Frames saved:", saved_count)
print("Saved at:", output_folder)
