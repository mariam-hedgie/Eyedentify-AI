const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const captureBtn = document.getElementById('captureBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const result = document.getElementById('result');
const capturedImg = document.getElementById('capturedImg');

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
  video.play();
});

video.addEventListener('loadedmetadata', () => {
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  drawOvalGuide();
});

function drawOvalGuide() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const centerX = overlay.width / 2;
  const centerY = overlay.height / 2.4;
  const radiusX = overlay.width / 3.5;
  const radiusY = overlay.height / 3;

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

function showResultOnLeft(message) {
  // Set the result message
  document.getElementById('resultMessage').innerHTML = message;

  // Hide intro content
  document.getElementById('introContent').style.display = 'none';

  // Show result content
  document.getElementById('resultContent').style.display = 'block';
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
      showResultOnLeft(`❌ Error: ${data.error}`);
    } else {
      const leftProb = Math.round(data.left_eye_prob * 100);
      const rightProb = Math.round(data.right_eye_prob * 100);

      const message = `
        👁️ <strong>Left Eye:</strong> ${leftProb}% chance of conjunctivitis<br>
        👁️ <strong>Right Eye:</strong> ${rightProb}% chance of conjunctivitis
      `;
      showResultOnLeft(message);
    }
  });
};


document.getElementById('tryAgainBtn').addEventListener('click', () => {
  // 1. Reset panels
  document.getElementById('resultContent').style.display = 'none';
  document.getElementById('introContent').style.display = 'block';

  // 2. Clear result message
  document.getElementById('resultMessage').innerHTML = '';

  // 3. Reset UI elements
  capturedImg.style.display = 'none';
  video.style.display = 'block';
  overlay.style.display = 'block';
  captureBtn.style.display = 'inline-block';
  analyzeBtn.style.display = 'none';

  // 4. Redraw oval guide (optional)
  drawOvalGuide();
});
