import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Step 1: Define GradCAM Class ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def save_activations(module, input, output):
            self.activations = output
        
        target_layer.register_forward_hook(save_activations)
        target_layer.register_backward_hook(save_gradients)

    def __call__(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # Global Average Pooling
        cam = (weights * activations).sum(dim=1, keepdim=True)  # Weighted sum
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize to [0,1]
        return cam

# --- Step 2: Load Model + Target Layer ---
model = models.resnet18(pretrained=True)
target_layer = model.layer4[1].conv2  # Last conv layer
gradcam = GradCAM(model, target_layer)

# --- Step 3: Load and Preprocess Image ---
img_path =  '/Users/mariamhusain/Desktop/eyedentify-ai/data/raw/healthy_eye/0.jpg' 
PIL_img = Image.open(img_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(PIL_img).unsqueeze(0)  # Shape: (1, 3, 224, 224)

# --- Step 4: Run Grad-CAM ---
target_class = 1  # 👈 Set this manually (0 = healthy, 1 = conjunctivitis)
heatmap = gradcam(input_tensor, class_idx=target_class)

# --- Step 5: Overlay Heatmap ---
# Convert original image to displayable format
img_np = np.array(PIL_img.resize((224, 224)))
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
overlay = np.uint8(0.5 * heatmap_color + 0.5 * img_np)

# --- Step 6: Save or Display ---
cv2.imwrite('gradcam_overlay.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print("Saved heatmap overlay to gradcam_overlay.jpg")

# Optional: Display inline (if running in notebook)
plt.imshow(overlay)
plt.axis('off')
plt.show()