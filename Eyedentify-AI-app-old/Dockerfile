# Use the official Python 3.11.1 slim image from Docker Hub
FROM python:3.11.1-slim

# Install system dependencies for OpenCV, Mediapipe, and Git
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container (for dependency installation)
COPY requirements.txt /app/

# Install the dependencies from the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Expose the port your app will run on (e.g., Flask default port 5000)
EXPOSE 5000

# Command to run your Flask app (or your app's main script)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

