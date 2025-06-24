# 👁️ Eyedentify-AI: Conjunctivitis Detection Using Deep Learning  
**🚧 Status:** In Progress — Expected Completion: July 31, 2025

**Eyedentify-AI** is a tool that classifies red eye (conjunctivitis) from patient-submitted images. It combines medical image preprocessing, signal-based blur detection, and deep learning-based classification.  
The goal is to build a fully functional and explainable AI workflow — from raw images to web deployment — tailored for real-world use.

---

## ✨ First 10 Days Highlights  

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
	•	🚫 Blur Filtering: Class-specific sharpness scores determine a dynamic threshold.
	•	📦 Crop Engine: A custom-trained YOLOv8 model detects eyes from patient images.
	•	🏷️ Label Mapping: Images are linked to labels; filtered outliers are logged and excluded.

---

## 🔭 Planned Next Steps

| Week | Focus Area                            |
|------|----------------------------------------|
| 2    | Train ResNet18 baseline classifier     |
| 3    | Add Grad-CAM visualizations            |
| 4    | Build Flask app (upload + webcam)      |
| 5    | Patient portal + symptom timeline      |
| 6    | Demo polish + final write-up           |

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

## 🔒 License
This is a private project under active development by **Mariam Husain** as part of an independent initiative to build deployable, explainable AI tools for healthcare.

**All rights reserved © 2025 Mariam Husain.**
Unauthorized use, copying, or distribution is strictly prohibited.

For academic use, licensing, or collaboration:
📩 Contact: [email](mailto:mariamh1121@gmail.com)


> This project is actively evolving. Logs, plots, and notebooks are structured for traceability and can be extended for medical imaging beyond conjunctivitis.
