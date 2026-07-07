# Gunicorn config for the Hugging Face Docker Space.
bind = "0.0.0.0:7860"

# Inference (MediaPipe + ResNet + Grad-CAM) can take tens of seconds on the
# free CPU tier; keep well above gunicorn's 30s default so requests aren't killed.
timeout = 240

# Each worker loads its own copy of the model into memory. Two workers gives
# a little concurrency while staying comfortable within the free tier's RAM.
workers = 2
