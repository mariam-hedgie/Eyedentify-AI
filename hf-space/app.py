"""Eyedentify-AI — single-service Flask app.

Serves the web UI AND runs the model in-process (MediaPipe eye detection +
ResNet18 classification + Grad-CAM++). No external model service, so there is
no cross-service call to time out. Designed to run as one Hugging Face
Docker Space.
"""

import base64
import io
import gc

import numpy as np
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import mediapipe as mp

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- Model + inference setup (loaded once at startup) ----------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
LEFT_EYE_IDS = [33, 133, 159, 160, 161, 144, 145, 153]
RIGHT_EYE_IDS = [362, 263, 386, 387, 388, 373, 374, 380]
PAD = 10

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("resnet18_weights.pth", map_location="cpu"))
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

cam = GradCAMPlusPlus(model, [model.layer4[-1]])


def extract_eye(image_np, landmarks, eye_ids):
    h, w, _ = image_np.shape
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_ids]
    x_coords, y_coords = zip(*coords)
    xmin, xmax = max(min(x_coords) - PAD, 0), min(max(x_coords) + PAD, w)
    ymin, ymax = max(min(y_coords) - PAD, 0), min(max(y_coords) + PAD, h)
    return image_np[ymin:ymax, xmin:xmax]


def run_cam_on_crop(crop):
    if crop.size == 0:
        return 0.0, ""

    img_np = cv2.resize(crop, (224, 224))
    img_uint8 = img_np.astype(np.uint8)
    tensor = preprocess(img_uint8).unsqueeze(0)

    with torch.no_grad():
        p = torch.sigmoid(model(tensor)).item()

    g = cam(input_tensor=tensor,
            targets=[BinaryClassifierOutputTarget(int(p > 0.5))])[0]
    cam_img = show_cam_on_image(img_np / 255.0, g, use_rgb=True)

    buf = io.BytesIO()
    Image.fromarray(cam_img.astype(np.uint8)).save(buf, format="PNG")
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return round(p, 2), f"data:image/png;base64,{base64_img}"


def infer(base64_str):
    img_bytes = base64.b64decode(base64_str.split(",")[1])
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_np = np.array(img)

    results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    landmarks = results.multi_face_landmarks[0].landmark
    left_crop = extract_eye(image_np, landmarks, LEFT_EYE_IDS)
    right_crop = extract_eye(image_np, landmarks, RIGHT_EYE_IDS)

    left_prob, left_gradcam = run_cam_on_crop(left_crop)
    right_prob, right_gradcam = run_cam_on_crop(right_crop)
    gc.collect()

    return {
        "left_eye_prob": float(left_prob),
        "left_gradcam": left_gradcam,
        "right_eye_prob": float(right_prob),
        "right_gradcam": right_gradcam,
    }


# --- Routes ----------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return "ok", 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        base64_image = data.get("image")
        if not isinstance(base64_image, str) or not base64_image.startswith("data:image"):
            return jsonify({"error": "Expected a base64 image data URL"}), 400

        return jsonify(infer(base64_image))
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    # Local dev only; production uses gunicorn (see Dockerfile / gunicorn.conf.py)
    app.run(host="0.0.0.0", port=7860, debug=False)
