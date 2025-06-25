import cv2
from pathlib import Path
from ultralytics import YOLO
import datetime

# Set up paths
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "runs" / "detect" / "train2" / "weights" / "best.pt"
save_base = project_root / "data" / "webcam_capture"
(save_base / "1eye").mkdir(parents=True, exist_ok=True)
(save_base / "2eyes").mkdir(parents=True, exist_ok=True)

# Load model
model = YOLO(str(model_path))

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default camera
img_id = 0

print("📷 Starting webcam... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO inference
    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)

    # Draw bounding boxes
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show preview
    cv2.imshow("Eye Detection - Press 's' to Save, 'q' to Quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(boxes) == 1:
            x1, y1, x2, y2 = boxes[0]
            crop = frame[y1:y2, x1:x2]
            filename = save_base / "1eye" / f"eye_{timestamp}.jpg"
            cv2.imwrite(str(filename), crop)
            print(f"✅ Saved 1 eye crop to {filename}")

        elif len(boxes) >= 2:
            for i, (x1, y1, x2, y2) in enumerate(boxes[:2]):
                crop = frame[y1:y2, x1:x2]
                filename = save_base / "2eyes" / f"eye{i+1}_{timestamp}.jpg"
                cv2.imwrite(str(filename), crop)
                print(f"✅ Saved eye {i+1} crop to {filename}")

        else:
            print("⚠️ No eyes detected to save")

    elif key == ord('q'):
        print("👋 Quitting...")
        break

cap.release()
cv2.destroyAllWindows()