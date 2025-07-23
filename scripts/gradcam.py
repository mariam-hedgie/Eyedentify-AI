# %%
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# %%
# === CONFIG ===
from pathlib import Path

# Safe for notebooks: use current working directory
ROOT_DIR = Path.cwd()  # or manually specify Path("/your/project/root")
DATA_DIR = ROOT_DIR / "data" / "filtered"
MODEL_WEIGHTS = Path("/Users/mariamhusain/Desktop/resnet18_weights.pth") #use local path

# Load image paths
healthy_imgs = list((DATA_DIR / "healthy_eye").glob("*.jpg"))
infected_imgs = list((DATA_DIR / "infected_eye").glob("*.jpg"))

image_paths = healthy_imgs[:10] + infected_imgs[:10]
LABELS = {0: "Healthy", 1: "Infected"}

# %%
# === Load model and weights ===
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location='cpu'))
model.eval()

# %%
# === Preprocess image ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# %%
# === Initialize Grad-CAM++ ===
from pytorch_grad_cam.utils.image import preprocess_image

target_layers = [model.layer4[-1]]  # Use the last layer before classification
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

# %%
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle(
    "Grad-CAM++ Sample Visualizations (10 Healthy + 10 Infected)",
    fontsize=16
)

# Explanatory line
fig.text(
    0.5,            # center horizontally
    0.90,           # just below the suptitle
    "The heatmap overlay (red regions) shows where the model is focusing its attention, "
    "and the percentage is our model’s confidence of prediction for each eye image.",
    ha='center',
    fontsize=12,
    color='gray'
)


# %%
for idx, img_path in enumerate(image_paths):
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    input_tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        logit = model(input_tensor).item()
        prob_inf = torch.sigmoid(torch.tensor(logit)).item()
        pred_label = 1 if prob_inf > 0.5 else 0

    if pred_label == 1:
        title = f"Infected ({prob_inf*100:.1f}% infected)"
    else:
        prob_healthy = (1 - prob_inf)
        title = f"Healthy ({prob_healthy*100:.1f}% healthy)"

    target = [BinaryClassifierOutputTarget(pred_label)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    ax = axes[idx // 5, idx % 5]
    ax.imshow(cam_image)
    ax.axis('off')
    
    ax.set_title(title, fontsize=10)

# %%
plt.tight_layout()
plt.subplots_adjust(top=0.85)
output_dir  = Path('.') / 'plots'  
output_file = output_dir / 'gradcam_visualizations.png'
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file)
plt.show()


