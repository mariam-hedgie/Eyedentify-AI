
# 👁️ Eyedentify.AI – Conjunctivitis Detection Web App

This is a real-time web-based screening tool that detects signs of **conjunctivitis (pink eye)** using a webcam and deep learning.  
It captures an image of your face, preprocesses it on the frontend, and uses a **cloud-hosted model** (via Hugging Face Spaces) to analyze each eye using **Grad-CAM-enhanced deep learning**.

---

## 🚀 Features

- 📸 Fullscreen camera interface with oval face guide
- 🧠 ResNet18 deep learning model with Grad-CAM++ visualizations
- 👁️ Eye detection powered by MediaPipe Face Mesh (via HF backend)
- ⚙️ Lightweight Flask frontend; inference offloaded to Hugging Face
- 🔐 All processing is user-controlled — no long-term image storage

---

## 🧰 Requirements

> This app runs entirely in your browser + server (no training required)

**Local Environment (Flask app only):**

- Python 3.10 or 3.11 (❗ Avoid Python 3.12+)
- macOS, Linux, or Windows
- Pip

---

## 🛠️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/eyedentify-ai-app.git
   cd eyedentify-ai-app
   ```

2. **(Optional) Create and activate virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install minimal dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask web server**

   ```bash
   python app.py
   ```

5. **Open in your browser:**

   [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧪 How It Works

1. The **browser** captures your image and sends it to the Flask server.
2. Flask pre-processes the image and forwards it to the **Gradio model endpoint** on Hugging Face:

   * Detects facial landmarks
   * Crops and analyzes both eyes
   * Runs ResNet18 on each crop
   * Generates Grad-CAM heatmaps
3. The results — probabilities and visual explanations — are returned to the browser.

---

## 🌐 Cloud Model Hosting (Hugging Face Spaces)

The AI model is deployed at:

```
https://huggingface.co/spaces/luckyjain1/eyedentify-ai-model
```

This Space:

* Uses MediaPipe for facial landmark detection
* Crops and preprocesses eye regions
* Runs predictions with ResNet18
* Generates Grad-CAM++ overlays
* Returns all results via JSON

---

## 📁 Folder Structure

```
eyedentify-ai-app/
├── app.py                 # Flask server
├── templates/
│   └── index.html         # Main UI
├── static/
│   └── script.js          # Camera + interaction logic
├── requirements.txt       # Trimmed dependencies
└── README.md              # You're reading it
```

## 🧠 Technical Highlights

* **Architecture**: Decoupled frontend/backend — lightweight Flask handles UI, heavy inference runs serverlessly on Hugging Face.
* **Explainability**: Grad-CAM++ provides visual attribution maps for transparency.
* **Efficiency**: Eye crops are extracted from a single face using MediaPipe's landmark mesh.

---

## 📸 Sample Output

> A Grad-CAM visualization will appear for each eye region, highlighting model attention on inflamed or discolored areas.

---