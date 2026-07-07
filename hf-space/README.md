---
title: Eyedentify AI
emoji: 👁️
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Eyedentify-AI

Conjunctivitis (red-eye) screening from a webcam image. Single-service app:
a Flask web UI that runs the model **in-process** — MediaPipe eye detection,
a ResNet18 classifier, and Grad-CAM++ explanations. No external model service.

## How it runs

Hugging Face builds the `Dockerfile` and serves the app on port 7860.
`git push` to this Space redeploys it.

- `app.py` — Flask routes (`/`, `/predict`, `/healthz`) + inference
- `templates/`, `static/` — the web UI
- `resnet18_weights.pth` — trained model weights (tracked with Git LFS)

For educational/demo use only; not a medical device.
