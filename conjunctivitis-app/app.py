from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import mediapipe as mp
from torchvision import models

app = Flask(__name__)

# === Load model ===
# Define architecture
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # binary classifier

# Load weights
model.load_state_dict(torch.load("resnet18_weights.pth", map_location=torch.device('cpu')))
model.eval()

# === MediaPipe setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# === Eye landmark indices ===
LEFT_EYE_IDS = [33, 133, 159, 160, 161, 144, 145, 153]
RIGHT_EYE_IDS = [362, 263, 386, 387, 388, 373, 374, 380]
PAD = 10

# === Preprocessing pipeline ===
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Step 1: Parse JSON
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    # Step 2: Decode base64 image
    try:
        img_data = data['image'].split(',')[1]
        image_pil = Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

    # Decode base64 image
    image_np = np.array(image_pil)
    h, w, _ = image_np.shape

    # Detect face
    results = face_mesh.process(image_np)
    if not results.multi_face_landmarks:
        return jsonify({"error": "No face detected"}), 400
    face_landmarks = results.multi_face_landmarks[0]

    def crop_eye(eye_ids):
        xs = [int(face_landmarks.landmark[i].x * w) for i in eye_ids]
        ys = [int(face_landmarks.landmark[i].y * h) for i in eye_ids]
        xmin, xmax = max(min(xs) - PAD, 0), min(max(xs) + PAD, w)
        ymin, ymax = max(min(ys) - PAD, 0), min(max(ys) + PAD, h)
        return image_np[ymin:ymax, xmin:xmax]

    # Crop both eyes
    left_eye_crop = crop_eye(LEFT_EYE_IDS)
    right_eye_crop = crop_eye(RIGHT_EYE_IDS)

    # Predict each
    def predict_eye(crop):
        input_tensor = preprocess(crop).unsqueeze(0)  # [1, 3, 224, 224]
        with torch.no_grad():
            output = model(input_tensor)
            return torch.sigmoid(output).item()

    left_prob = predict_eye(left_eye_crop)
    right_prob = predict_eye(right_eye_crop)

    return jsonify({
        "left_eye_prob": round(left_prob, 2),
        "right_eye_prob": round(right_prob, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
