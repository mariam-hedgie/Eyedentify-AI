# 👁️ Eyedentify-AI: Conjunctivitis Detection Using Deep Learning  
**🚧 Status:** In Progress — Expected Completion: July 31, 2025

**Eyedentify-AI** is a tool that classifies red eye (conjunctivitis) from patient-submitted images. It combines medical image preprocessing, signal-based blur detection, and deep learning-based classification.  
The goal is to build a fully functional and explainable AI workflow — from raw images to web deployment — tailored for real-world use.

---

## ✨ First 10 Days Highlights  

| Component                        | Skills Applied                      |
|----------------------------------|--------------------------------------|
| GitHub repo + folder setup       | Version control, modular pipeline design     |
| Image resizing (224×224)         | OpenCV, consistent model input prep  |
| FFT-based blur detection         | NumPy, frequency domain filtering    |
| Dynamic thresholds by class      | Distribution-aware logic, automation |
| Label mapping & blur logging     | pandas, data hygiene                 |
| Sharpness histograms             | matplotlib, interpretability         |

---

## 🧠 Current Logic

- All images are resized and pre-cleaned.
- Blur detection is done using **Fast Fourier Transform (FFT)**.
- Class-specific sharpness scores are analyzed to dynamically threshold and filter the bottom 5% of images.
- Clean images are mapped to labels; blurry or corrupt ones are automatically logged.

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

## 🛠️ Stack

Python, OpenCV, NumPy, pandas, Matplotlib  
Planned: PyTorch, Grad-CAM, Flask

---

## 🔒 License
This project is part of an independent initiative to build deployable AI tools in healthcare.

This project is not open source. All rights reserved © 2025 Mariam Husain.

Unauthorized use, distribution, or reproduction of any part of this repository is strictly prohibited.

For licensing, academic use, or collaboration inquiries, please contact me.


> This project is actively evolving. Logs, plots, and notebooks are structured for traceability and can be extended for medical imaging beyond conjunctivitis.
