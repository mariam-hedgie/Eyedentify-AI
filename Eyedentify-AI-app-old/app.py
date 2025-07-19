from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import mediapipe as mp
import requests
import os

app = Flask(__name__)

# Set Hugging Face Inference API URL
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/luckyjain1/eyedentify-ai"

# === MediaPipe setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# === Eye landmark indices ===
LEFT_EYE_IDS = [33, 133, 159, 160, 161, 144, 145, 153]
RIGHT_EYE_IDS = [362, 263, 386, 387, 388, 373, 374, 380]
PAD = 10

# === Preprocessing pipeline ===
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.transpose(image, (2, 0, 1))  # C, H, W
    image = image / 255.0  # Normalize to [0, 1]
    return image.tolist()  # Convert to list for JSON

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

    # Convert to numpy array
    image_np = np.array(image_pil)
    h, w, _ = image_np.shape

    # Detect face using MediaPipe
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_np)

    if not results.multi_face_landmarks:
        return jsonify({"error": "No face detected"}), 400

    face_landmarks = results.multi_face_landmarks[0]

    # Crop eyes based on face landmarks
    def crop_eye(eye_ids):
        xs = [int(face_landmarks.landmark[i].x * w) for i in eye_ids]
        ys = [int(face_landmarks.landmark[i].y * h) for i in eye_ids]
        xmin, xmax = max(min(xs) - PAD, 0), min(max(xs) + PAD, w)
        ymin, ymax = max(min(ys) - PAD, 0), min(max(ys) + PAD, h)
        return image_np[ymin:ymax, xmin:xmax]

    left_eye_crop = crop_eye(LEFT_EYE_IDS)
    right_eye_crop = crop_eye(RIGHT_EYE_IDS)

    # Preprocess image for Hugging Face model
    left_eye_input = preprocess_image(Image.fromarray(left_eye_crop))
    right_eye_input = preprocess_image(Image.fromarray(right_eye_crop))

    # Step 3: Send request to Hugging Face API
    def get_model_prediction(image_input):
        response = requests.post(
            HUGGINGFACE_API_URL,
            json={"inputs": image_input}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error from Hugging Face: {response.text}"}

    left_prediction = get_model_prediction(left_eye_input)
    right_prediction = get_model_prediction(right_eye_input)

    # Check if there is an error in predictions
    if "error" in left_prediction:
        return jsonify(left_prediction), 400
    if "error" in right_prediction:
        return jsonify(right_prediction), 400

    # Extract probabilities from model response
    left_prob = left_prediction.get("label", 0)
    right_prob = right_prediction.get("label", 0)

    # Placeholder Grad-CAM function (can be adjusted for Hugging Face integration later)
    def generate_placeholder_gradcam():
        # This part can be replaced with Grad-CAM logic if the model supports it
        return None, None  # Placeholder for now

    left_gradcam, right_gradcam = generate_placeholder_gradcam()

    # Return results
    return jsonify({
        "left_eye_prob": round(left_prob, 2),
        "right_eye_prob": round(right_prob, 2),
        "left_gradcam": left_gradcam,
        "right_gradcam": right_gradcam
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
