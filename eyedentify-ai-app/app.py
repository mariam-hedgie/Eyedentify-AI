from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image
from gradio_client import Client

app = Flask(__name__)

# The model runs on a HuggingFace Space that sleeps after ~48h idle.
# Reuse a single client instead of rebuilding it (and re-fetching the Space
# config) on every request.
HF_SPACE = "luckyjain1/eyedentify-ai-model"
_client = None


def get_client():
    global _client
    if _client is None:
        # Bump the underlying httpx timeout: the first inference after the
        # Space wakes can be slow (model loading), and the default read
        # timeout would otherwise abort it.
        _client = Client(HF_SPACE, httpx_kwargs={"timeout": 150})
    return _client


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    try:
        base64_image = data['image']
        if not isinstance(base64_image, str):
            return jsonify({"error": "Image must be a base64-encoded string"}), 400

        if not base64_image.startswith("data:image"):
            return jsonify({"error": "Expected full base64 image string with data:image/... prefix"}), 400

        # Strip the header and decode
        header, encoded = base64_image.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        # Open image and convert to RGB
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Re-encode to base64 JPEG
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        reencoded_bytes = buffered.getvalue()
        reencoded_base64 = base64.b64encode(reencoded_bytes).decode('utf-8')
        final_base64_image = f"data:image/jpeg;base64,{reencoded_base64}"

        # Gradio client call (send clean RGB image)
        try:
            client = get_client()
            hf_result = client.predict(
                final_base64_image,
                api_name="/predict"
            )
        except Exception as e:
            # Most common cause: the HF Space is asleep/cold-starting.
            # Drop the cached client so the next request rebuilds it.
            global _client
            _client = None
            return jsonify({
                "error": "The model service is starting up (it sleeps when "
                         "idle). Please wait a minute and try again."
            }), 503

        return jsonify(hf_result)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

import socket

def get_free_port(default=5000):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", default))
            return default
    except OSError:
        # If default port is in use, pick any free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

if __name__ == "__main__":
    port = get_free_port()
    print(f"Running on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port, threaded=False)
