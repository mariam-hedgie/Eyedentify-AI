# 👁️ Eyedentify-AI: Conjunctivitis Detection Using Deep Learning  
**🚧 Status:** In Progress — Expected Completion: End July, 2025

**Eyedentify-AI** is a tool that classifies red eye (conjunctivitis) from patient-submitted images. It combines medical image preprocessing, signal-based blur detection, and deep learning-based classification.  
The goal is to build a fully functional and explainable AI workflow — from raw images to web deployment — tailored for real-world use.

---

## ✨ First 1 month Highlights  

| Component                        | Skills Applied                      |
|----------------------------------|--------------------------------------|
| GitHub repo + folder setup       | Version control, modular pipeline design     |
| Virtual environment setup        | Dependency management, reproducibility |
| Image resizing (224×224)         | OpenCV, model input preparation  |
| FFT-based blur detection         | NumPy, frequency domain analysis    |
| Sharpness histogram visualization | Matplotlib, exploratory data analysis |
| Dynamic filtering by class distribution      | Distribution-aware logic, automation |
| Label mapping & blur logging     | pandas, data hygiene                 |
| YOLOv8 eye detector: custom-trained             | Roboflow labeling, PyTorch training, inference logic        |

---

## 🧠 Current Logic

•	🔍 Preprocessing: Images are resized, normalized, and passed through an FFT-based blur detector.
<br>
•	🚫 Blur Filtering: Class-specific sharpness scores determine a dynamic threshold.
<br>
•	📦 Crop Engine: A custom-trained YOLOv8 model detects eyes from patient images.
<br>
•	🏷️ Label Mapping: Images are linked to labels; filtered outliers are logged and excluded.

---

## 🔭 Planned Next Steps

• Train ResNet18 baseline classifier 
• Add Grad-CAM visualizations
• Build Flask app (upload + webcam)     
• Patient portal + symptom timeline      
• Demo polish + final write-up           

---

## 🛠️ Tech Stack

| Layer | Tech Stack |
| ----- | -----------|
| Preprocessing | Python, OpenCV, NumPy, pandas, Matplotlib |
| Detection | YOLOv8 (Ultralytics), Roboflow annotations |
| Classification | PyTorch, Torchvision |
| Explainability | Grad-CAM (planned) |
| Web Interface | Flask (planned), SQLite or JSON-based state tracking |

---

# My Conjunctivitis App

Here’s the end‐to‐end screening pipeline:

```mermaid
flowchart TD
  A["Capture webcam image"] --> B["Click \"Analyze\""]
  B --> C["MediaPipe detects eye region"]
  C --> D["Preprocess: resize to 224×224 & normalize"]
  D --> E["ResNet18 inference → P(infected)"]

  E --> F{"P(infected) ≥ 60%?"}
  F -- Yes --> G["Likely Conjunctivitis"]
  F -- No  --> H{"P(infected) ≥ 40%?"}
  H -- Yes --> I["Near threshold: monitor or consult"]
  H -- No  --> J["Likely Normal"]

  E --> K["Generate Grad-CAM++ heatmap"]
  K --> L["Overlay heatmap & display results"]

  L --> M["Try Again ▶️"]
  L --> N["Learn More 🔍"]
  

## 🔒 License
This is a private project under active development by **Mariam Husain** as part of an independent initiative to build deployable, explainable AI tools for healthcare.

**All rights reserved © 2025 Mariam Husain.**
Unauthorized use, copying, or distribution is strictly prohibited.

For academic use, licensing, or collaboration:
📩 [Contact Me](mailto:mariamh1121@gmail.com)


> This project is actively evolving. Logs, plots, and notebooks are structured for traceability and can be extended for medical imaging beyond conjunctivitis.
