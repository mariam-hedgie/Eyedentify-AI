# 👁️ Conjunctivitis Detection Web App

This is a Flask-based web application that uses a webcam to capture a photo of a user's face, detects the eyes using MediaPipe, and runs a deep learning model (ResNet18) to predict the probability of conjunctivitis in the left and right eyes.

---

## 🚀 Features

* Fullscreen camera with oval face guide
* Capture and preview image before analysis
* Detect face and crop both eyes using MediaPipe
* Run inference using a PyTorch model (ResNet18)
* Display left and right eye conjunctivitis probabilities

---

## 🧰 Requirements

* Python 3.10 or 3.11 (do **not** use 3.12+)
* macOS (Intel)
* Pip

---

## 🛠️ Setup Instructions (Intel Mac)

1. **Clone this repository**

   ```bash
   git clone https://github.com/your-username/conjunctivitis-app.git
   cd conjunctivitis-app
   ```

2. **Create and activate virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download your trained model**

   Place your model weights file (e.g., `resnet18_weights.pth`) in the project folder.

   Make sure `app.py` matches your model structure:

   ```python
   from torchvision import models
   model = models.resnet18(pretrained=False)
   model.fc = torch.nn.Linear(model.fc.in_features, 1)
   model.load_state_dict(torch.load("resnet18_weights.pth", map_location=torch.device('cpu')))
   ```

5. **Run the Flask app**

   ```bash
   python app.py
   ```

6. **Visit the app**
   Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧪 How It Works

* The frontend accesses your webcam and draws an oval where your face should go.
* When you click 📸, it captures an image and sends it to the backend.
* The backend:

  * Uses MediaPipe to detect eye landmarks
  * Crops left and right eye regions
  * Preprocesses each crop
  * Runs them through a PyTorch model
  * Returns the probability of conjunctivitis for each eye
* The frontend displays the results clearly on screen.

---

## 🧾 Dependencies (from `requirements.txt`)

```
Flask
torch
torchvision
torchaudio
mediapipe
numpy
Pillow
opencv-python
```

---

## 📦 Folder Structure

```
conjunctivitis-app/
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── script.js
├── resnet18_weights.pth
├── requirements.txt
└── README.md
```

---

## 🧠 Notes

* This version uses CPU for inference. For GPU, modify the model loading and use `map_location='cuda'` if available.
* Currently uses only MediaPipe's first detected face.
* TorchScript or ONNX conversion is supported if you plan to port to mobile.

---

## 📬 Contact

Built by \[Your Name]. Feel free to reach out with questions or suggestions!
