import cv2
import torch
from pathlib import Path
from datetime import datetime

# === Load YOLOv8 model ===
model_path = "models/yolov8_eye.pt"  # Update if needed
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# === Haar Cascade for face detection ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Setup webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("📷 Press 'q' to quit.")
save_dir = Path("data/split/webcam_capture")
save_dir.mkdir(parents=True, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not read correctly.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        eye_region = frame[y:y + h // 2, x:x + w]  # upper half of face
        results = model(eye_region)

        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(eye_region, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Save crop
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            crop = eye_region[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(str(save_dir / f"eye_{timestamp}.jpg"), crop)

        # Display the detected region
        frame[y:y + h // 2, x:x + w] = eye_region
        break  # Only use first face for now

    cv2.imshow("YOLO Eye Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()