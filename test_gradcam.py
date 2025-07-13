import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam.grad_cam_plus_plus import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# === CONFIG ===
IMAGE_PATH = "/Users/mariamhusain/Desktop/Eyedentify-AI/data/filtered/healthy_eye/825.jpg"
MODEL_WEIGHTS = "/Users/mariamhusain/Desktop/resnet18_weights.pth"
LABELS = {0: "Healthy", 1: "Infected"}

# === Load model and weights ===
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location='cpu'))
model.eval()

# === Preprocess image ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img_pil = Image.open(IMAGE_PATH).convert("RGB")
img_np = np.array(img_pil.resize((224, 224))) / 255.0
input_tensor = transform(img_pil).unsqueeze(0)

# === Grad-CAM++ Setup ===
target_layer = model.layer4[-1]
cam = GradCAMPlusPlus(model=model, target_layers=[target_layer], use_cuda=False)

# === Model Prediction ===
with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()
    pred_label = 1 if prob > 0.5 else 0

# === Generate CAM mask ===
target = [BinaryClassifierOutputTarget(pred_label)]
grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]

# === Overlay on original image ===
cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

# === Display ===
plt.imshow(cam_image)
plt.axis('off')
plt.title(f"{LABELS[pred_label]} ({prob:.2f})")
plt.tight_layout()
plt.show()