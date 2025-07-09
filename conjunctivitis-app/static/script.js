const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const captureBtn = document.getElementById('captureBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const result = document.getElementById('result');
const capturedImg = document.getElementById('captured');

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
});

video.addEventListener('loadedmetadata', () => {
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  drawOvalGuide();
});

function drawOvalGuide() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const centerX = overlay.width / 2;
  const centerY = overlay.height / 2.2;
  const radiusX = overlay.width / 6;
  const radiusY = overlay.height / 3.5;

  ctx.fillStyle = "white";
  for (let angle = 0; angle < 360; angle += 12) {
    const rad = angle * Math.PI / 180;
    const x = centerX + radiusX * Math.cos(rad);
    const y = centerY + radiusY * Math.sin(rad);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();
  }
}

// CAPTURE button logic
captureBtn.onclick = () => {
  const tempCanvas = document.createElement('canvas');
  const tCtx = tempCanvas.getContext('2d');
  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;

  // Mirror the image before capture to match display
  tCtx.translate(tempCanvas.width, 0);
  tCtx.scale(-1, 1);
  tCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

  // Freeze UI
  const imageDataURL = tempCanvas.toDataURL('image/png');
  capturedImg.src = imageDataURL;
  capturedImg.style.display = 'block';
  video.style.display = 'none';
  overlay.style.display = 'none';
  captureBtn.style.display = 'none';
  analyzeBtn.style.display = 'inline-block';
};

// ANALYZE button logic
analyzeBtn.onclick = () => {
  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: capturedImg.src })
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      result.innerText = data.error;
    } else {
      document.getElementById('leftResult').innerText =
        `👁️ Left Eye: ${Math.round(data.left_eye_prob * 100)}% probability of conjunctivitis`;
      document.getElementById('rightResult').innerText =
        `👁️ Right Eye: ${Math.round(data.right_eye_prob * 100)}% probability of conjunctivitis`;
    }
  });
};

