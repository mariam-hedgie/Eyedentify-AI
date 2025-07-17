const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const captureBtn = document.getElementById('captureBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const result = document.getElementById('result');
const capturedImg = document.getElementById('capturedImg');
const learnBtn = document.getElementById('learnBtn');
const gradcamContent = document.getElementById('gradcamContent');
const leftGradcam = document.getElementById('leftGradcam');
const rightGradcam = document.getElementById('rightGradcam');


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
      const avgProb = (leftProb + rightProb) / 2;

      let label = '';
      let advice = '';

      if (avgProb >= 60) {
        label = 'Likely Conjunctivitis';
        advice = '⚠️ Please consult an eye specialist.';
      } else if (avgProb >= 40) {
        label = 'Uncertain';
        advice = '🧐 You’re close to the threshold — monitor symptoms or consult if unsure.';
      } else {
        label = 'Likely Normal';
        advice = '✅ No strong signs of conjunctivitis. Keep eyes clean and healthy!';
      }
      
      const message = `
        
        <p style="font-size: 18px; color: ${avgProb >= 60 ? '#dc2626' : avgProb >= 40 ? '#d97706' : '#16a34a'};">
  ${label}
        </p>
        <p>${advice}</p>
      `;
      showResultOnLeft(message);

      // Save Grad-CAM images for later display
      leftGradcam.src = `data:image/png;base64,${data.left_gradcam}`;
      rightGradcam.src = `data:image/png;base64,${data.right_gradcam}`;

      function getEyeAdvice(prob, side) {
        if (prob >= 60) return `${side} Eye: High attention on redness — 📍 consult a doctor.`;
        if (prob >= 40) return `${side} Eye: Mild signs — 🧐 monitor symptoms.`;
        return `${side} Eye: Low risk — ✅ no concerning signs.`;
      }
      
      document.getElementById('leftExplanation').textContent =
        `🔬 ${leftProb}% chance. ` + getEyeAdvice(leftProb, "Left");
      
      document.getElementById('rightExplanation').textContent =
        `🔬 ${rightProb}% chance. ` + getEyeAdvice(rightProb, "Right");
    }
  });
};


document.getElementById('tryAgainBtn').addEventListener('click', () => {
  // 1. Reset panels
  document.getElementById('resultContent').style.display = 'none';
  document.getElementById('introContent').style.display = 'block';

  // 2. Clear result message
  document.getElementById('resultMessage').innerHTML = '';

  // 3. resert grad cam
  gradcamContent.style.display = 'none';
  learnBtn.style.display = 'inline-block';

  // 4. Reset UI elements
  capturedImg.style.display = 'none';
  video.style.display = 'block';
  overlay.style.display = 'block';
  captureBtn.style.display = 'inline-block';
  analyzeBtn.style.display = 'none';

  // 5. Redraw oval guide (optional)
  drawOvalGuide();
});

learnBtn.onclick = () => {
  gradcamContent.style.display = 'block'; // show grad cam container
  learnBtn.style.display = 'none'; // hide button after click
};

