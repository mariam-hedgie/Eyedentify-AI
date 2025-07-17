from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import mediapipe as mp
from torchvision import models
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import requests
import sys
from tqdm import tqdm

app = Flask(__name__)

import os

def download_with_progress(url, destination):
    if os.path.exists(destination):
        print("Weights already downloaded.")
        return

    print(f"Downloading model weights to {destination}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    with open(destination, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

#earlier weights loading without download link
#weights_path = os.path.join(os.path.dirname(__file__), "resnet18_weights.pth")
#model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

# Define model architecture
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # binary classifier

# Google Drive direct download link
model_url = "https://drive.google.com/uc?export=download&id=15fLW_oFGFnqYSMa4BfGnXjeYIWo2bNr_"
weights_path = os.path.join(os.path.dirname(__file__), "resnet18_weights.pth")

download_with_progress(model_url, weights_path)

# Load weights
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
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
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
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

    def generate_gradcam_image(model,eye_crop_np, predicted_label):

        # Keep a version for overlay
        img_np = cv2.resize(eye_crop_np, (224, 224)) / 255.0

        # Feed the same original image into transform for model input
        img_uint8 = (img_np*255).astype(np.uint8)
        input_tensor = preprocess(img_uint8).unsqueeze(0)

        # Set up Grad-CAM++
        target_layers = [model.layer4[-1]]
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

        # Create target
        target = [BinaryClassifierOutputTarget(predicted_label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]

        # Apply CAM
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        cam_pil = Image.fromarray(cam_image)

        return cam_pil
    
    def generate_gradcam():

        left_gradcam = generate_gradcam_image(model, left_eye_crop, int(left_prob > 0.5))
        right_gradcam = generate_gradcam_image(model, right_eye_crop, int(right_prob > 0.5))
        return left_gradcam, right_gradcam


    # Generate Grad-CAM image
    left_cam, right_cam = generate_gradcam()

    # Convert to base64
    buffer_left = BytesIO()
    left_cam.save(buffer_left, format = "PNG")
    encoded_left = base64.b64encode(buffer_left.getvalue()).decode("utf-8")

    buffer_right = BytesIO()
    right_cam.save(buffer_right, format = "PNG")
    encoded_right = base64.b64encode(buffer_right.getvalue()).decode("utf-8")

    return jsonify({
    "left_eye_prob": round(left_prob, 2),
    "right_eye_prob": round(right_prob, 2),
    "left_gradcam": encoded_left,
    "right_gradcam": encoded_right
    })

if __name__ == '__main__':
    app.run(debug=True)
