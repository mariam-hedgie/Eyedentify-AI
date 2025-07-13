import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# === CONFIG ===
IMAGE_PATH = "/Users/mariamhusain/Desktop/eyedentify-ai/data/filtered/healthy_eye/0.jpg"
MODEL_WEIGHTS = "resnet18_weights.pth"
LABELS = {0: "Healthy", 1: "Infected"}  # your label meanings

# === Load model and weights ===
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification head
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location='cpu'))
model.eval()

# === Select target layer for Grad-CAM ===
target_layer = model.layer4[-1]  # Last conv layer in ResNet18

# === Load and preprocess the image ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img_pil = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img_pil).unsqueeze(0)  # [1, 3, 224, 224]

# === Hooks to capture activation and gradient ===
gradients = []
activations = []

def save_activation(module, input, output):
    activations.append(output.detach())

def save_gradient(module, grad_input, grad_output):
    gradients.append(grad_output[0].detach())

# Register hooks
handle_act = target_layer.register_forward_hook(save_activation)
handle_grad = target_layer.register_backward_hook(save_gradient)

# === Forward pass ===
output = model(input_tensor)
prob = torch.sigmoid(output).item()
pred_label = 1 if prob > 0.5 else 0

print(f"Predicted Class: {LABELS[pred_label]} | Confidence: {prob:.4f}")

# === Backward pass to get gradients w.r.t. prediction ===
model.zero_grad()
output.backward(torch.ones_like(output))

# === Compute Grad-CAM ===
grad = gradients[0][0]         # [C, H, W]
act = activations[0][0]        # [C, H, W]
weights = torch.mean(grad, dim=(1, 2))  # Global average pooling
cam = torch.sum(weights[:, None, None] * act, dim=0)  # Weighted sum
cam = torch.relu(cam).numpy()
cam = cv2.resize(cam, (224, 224))
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# === Overlay heatmap ===
img_np = np.array(img_pil.resize((224, 224))) / 255.0
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = 0.5 * heatmap / 255.0 + 0.5 * img_np

# === Display ===
plt.imshow(overlay)
plt.axis('off')
plt.title(f"{LABELS[pred_label]} ({prob:.2f})")
plt.tight_layout()
plt.show()

# === Clean up hooks ===
handle_act.remove()
handle_grad.remove()