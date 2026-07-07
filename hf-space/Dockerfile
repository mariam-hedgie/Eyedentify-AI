FROM python:3.11.1-slim

# System deps for OpenCV / MediaPipe, plus git for the grad-cam VCS install
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first so pip doesn't pull the huge CUDA wheels
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Hugging Face Docker Spaces route external traffic to port 7860
EXPOSE 7860

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
