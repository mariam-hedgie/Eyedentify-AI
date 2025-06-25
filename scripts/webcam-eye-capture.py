import cv2
import torch
from datetime import datetime
from pathlib import Path

# Load trained YOLO model (change path if needed)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov8_eye.pt', force_reload=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam.")
    exit()

print("📸 Running Eye Detector. Press 'q' to quit.")

save_path = Path("data/split/webcam_capture")
save_path.mkdir(parents=True, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = frame[y:y + h // 2, x:x + w]  # upper half (eye region)
        results = model(roi)

        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

            eye_crop = roi[y1:y2, x1:x2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            if eye_crop.size > 0:
                cv2.imwrite(str(save_path / f"eye_{timestamp}.jpg"), eye_crop)

        frame[y:y + h // 2, x:x + w] = roi
        break  # only process first face

    cv2.imshow("YOLOv8 Eye Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()