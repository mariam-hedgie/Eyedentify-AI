from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

from gradio_client import Client

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

        # Use Gradio client
        client = Client("luckyjain1/eyedentify-ai-model")
        hf_result = client.predict(
            base64_str=base64_image,
            api_name="/predict"
        )

        return jsonify(hf_result)

    except Exception as e:
        return jsonify({"error": f"Gradio client failed: {str(e)}"}), 500

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
