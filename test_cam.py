import cv2

print("Opening camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera NOT opening")
    exit()

print("Camera opened")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame")
        continue

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

