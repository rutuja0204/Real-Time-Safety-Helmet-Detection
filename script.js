'use strict';

// ── DOM references ────────────────────────────────────────────────────────────
const liveTime          = document.getElementById('liveTime');
const startButtons      = [document.getElementById('startCameraBtn'), document.getElementById('startCameraBtnMain')];
const stopButtons       = [document.getElementById('stopCameraBtn'),  document.getElementById('stopCameraBtnMain')];
const simulateBtn       = document.getElementById('simulateBtn');
const demoDetectionsBtn = document.getElementById('demoDetectionsBtn');
const resetStatsBtn     = document.getElementById('resetStatsBtn');
const streamImage       = document.getElementById('streamImage');
const previewImage      = document.getElementById('previewImage');
const imageUpload       = document.getElementById('imageUpload');
const videoUpload       = document.getElementById('videoUpload');
const cameraModeLabel   = document.getElementById('cameraModeLabel');
const eventTableBody    = document.getElementById('eventTableBody');
const cameraStatus      = document.getElementById('cameraStatus');
const modelStatus       = document.getElementById('modelStatus');
const systemMessage     = document.getElementById('systemMessage');
const connectionBadge   = document.getElementById('connectionBadge');

const totalDetections = document.getElementById('totalDetections');
const helmetCount     = document.getElementById('helmetCount');
const noHelmetCount   = document.getElementById('noHelmetCount');
const complianceScore = document.getElementById('complianceScore');
const helmetPercent   = document.getElementById('helmetPercent');
const noHelmetPercent = document.getElementById('noHelmetPercent');
const helmetBar       = document.getElementById('helmetBar');
const noHelmetBar     = document.getElementById('noHelmetBar');

let statsPoller = null;

// ── Clock ─────────────────────────────────────────────────────────────────────
function updateClock() {
  liveTime.textContent = new Date().toLocaleTimeString();
}
setInterval(updateClock, 1000);
updateClock();

// ── System message ────────────────────────────────────────────────────────────
function setSystemMessage(message, isError = false) {
  if (!systemMessage) return;
  systemMessage.textContent = message;
  systemMessage.classList.toggle('error-text', isError);
}

// ── Button loading state ──────────────────────────────────────────────────────
/**
 * Temporarily disable a button and show a loading label so the user knows
 * the action is in-flight. Returns a restore function.
 */
function setButtonLoading(btn, loadingText) {
  if (!btn) return () => {};
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = loadingText;
  return () => {
    btn.disabled = false;
    btn.textContent = original;
  };
}

// ── Render helpers ────────────────────────────────────────────────────────────
function renderStats(stats) {
  const total     = stats.total     || 0;
  const helmet    = stats.helmet    || 0;
  const noHelmet  = stats.no_helmet || 0;
  const compliance = stats.compliance || 0;
  const violation  = total ? 100 - compliance : 0;

  totalDetections.textContent = total;
  helmetCount.textContent     = helmet;
  noHelmetCount.textContent   = noHelmet;
  complianceScore.textContent = `${compliance}%`;
  helmetPercent.textContent   = `${compliance}%`;
  noHelmetPercent.textContent = `${violation}%`;
  helmetBar.style.width       = `${compliance}%`;
  noHelmetBar.style.width     = `${violation}%`;

  modelStatus.textContent  = stats.model_loaded ? 'Loaded'  : 'Missing';
  modelStatus.className    = `pill ${stats.model_loaded ? 'good' : 'danger'}`;
  cameraStatus.textContent = stats.running ? 'Running' : 'Stopped';
  cameraStatus.className   = `pill ${stats.running ? 'good' : 'danger'}`;

  connectionBadge.textContent  = stats.running ? 'Backend Live'          : 'Backend Ready';

  if (stats.running && stats.is_video_file) {
    cameraModeLabel.textContent = `Video: ${stats.video_filename}`;
  } else {
    cameraModeLabel.textContent = stats.running ? 'YOLO Webcam Detection' : 'Detection Idle';
  }

  renderEventRows(stats.recent_events || []);

  if (!stats.model_loaded && stats.model_error) {
    setSystemMessage(stats.model_error, true);
  }
}

function renderEventRows(events) {
  eventTableBody.innerHTML = '';

  if (!events.length) {
    eventTableBody.innerHTML = `
      <tr>
        <td colspan="4" class="empty-row">No detections yet. Start the webcam or upload an image.</td>
      </tr>`;
    return;
  }

  events.forEach((event) => {
    const row       = document.createElement('tr');
    const pillClass = event.status === 'Helmet' ? 'good' : 'danger';
    row.innerHTML = `
      <td>${event.time}</td>
      <td>${event.zone}</td>
      <td><span class="table-pill ${pillClass}">${event.status}</span></td>
      <td>${event.confidence}%</td>`;
    eventTableBody.appendChild(row);
  });
}

// ── Stats polling ─────────────────────────────────────────────────────────────
async function fetchStats() {
  try {
    const response = await fetch('/stats');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const stats = await response.json();
    renderStats(stats);
  } catch {
    setSystemMessage('Could not connect to backend. Run python app.py first.', true);
  }
}

function startPolling() {
  if (statsPoller) clearInterval(statsPoller);
  statsPoller = setInterval(fetchStats, 1000);
}

// ── Camera controls ───────────────────────────────────────────────────────────
async function startCamera(triggerBtn) {
  const restore = setButtonLoading(triggerBtn, 'Starting…');
  try {
    const response = await fetch('/start_camera', { method: 'POST' });
    const result   = await response.json();
    if (!result.ok) {
      setSystemMessage(result.message, true);
      return;
    }

    previewImage.style.display = 'none';
    streamImage.style.display  = 'block';
    streamImage.src            = `/video_feed?ts=${Date.now()}`;
    setSystemMessage(result.message);
    startPolling();
    fetchStats();
  } catch {
    setSystemMessage('Could not start detection backend.', true);
  } finally {
    restore();
  }
}

async function stopCamera(triggerBtn) {
  const restore = setButtonLoading(triggerBtn, 'Stopping…');
  try {
    const response = await fetch('/stop_camera', { method: 'POST' });
    const result   = await response.json();
    streamImage.src = `/video_feed?ts=${Date.now()}`;
    setSystemMessage(result.message);
    fetchStats();
  } catch {
    setSystemMessage('Could not stop detection backend.', true);
  } finally {
    restore();
  }
}

async function resetStats(triggerBtn) {
  const restore = setButtonLoading(triggerBtn, 'Resetting…');
  try {
    const response = await fetch('/reset_stats', { method: 'POST' });
    const result   = await response.json();
    setSystemMessage(result.message);
    fetchStats();
  } catch {
    setSystemMessage('Could not reset stats.', true);
  } finally {
    restore();
  }
}

// ── Event listeners ───────────────────────────────────────────────────────────
startButtons.forEach(btn => btn?.addEventListener('click', () => startCamera(btn)));
stopButtons.forEach(btn  => btn?.addEventListener('click', () => stopCamera(btn)));

simulateBtn?.addEventListener('click', fetchStats);

demoDetectionsBtn?.addEventListener('click', async () => {
  await fetchStats();
  setSystemMessage('Backend status refreshed.');
});

resetStatsBtn?.addEventListener('click', () => resetStats(resetStatsBtn));

// ── Image upload ──────────────────────────────────────────────────────────────
imageUpload?.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  // Validate client-side before sending
  const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'];
  if (!allowedTypes.includes(file.type)) {
    setSystemMessage('Unsupported file type. Please upload a JPG, PNG, BMP, or WebP image.', true);
    imageUpload.value = '';
    return;
  }

  const MAX_MB = 16;
  if (file.size > MAX_MB * 1024 * 1024) {
    setSystemMessage(`File too large. Maximum allowed size is ${MAX_MB} MB.`, true);
    imageUpload.value = '';
    return;
  }

  const formData = new FormData();
  formData.append('image', file);

  try {
    setSystemMessage('Running image detection…');
    const response = await fetch('/detect_image', { method: 'POST', body: formData });

    // Handle 413 Payload Too Large from server
    if (response.status === 413) {
      setSystemMessage('File rejected by server: too large.', true);
      return;
    }

    const result = await response.json();
    if (!result.ok) {
      setSystemMessage(result.message, true);
      return;
    }

    previewImage.src            = `data:image/jpeg;base64,${result.image_base64}`;
    previewImage.style.display  = 'block';
    streamImage.style.display   = 'none';
    cameraModeLabel.textContent = 'Uploaded Image Detection';

    if (result.detections.length) {
      const summary = result.detections
        .map(item => `${item.label} (${item.confidence}%)`)
        .join(', ');
      setSystemMessage(`Detection complete: ${summary}`);
    } else {
      setSystemMessage('Image processed — no objects detected.');
    }

    fetchStats();
  } catch {
    setSystemMessage('Image detection failed. Is the backend running?', true);
  } finally {
    // Reset so the same file can be re-uploaded if needed
    imageUpload.value = '';
  }
});

// ── Video upload ──────────────────────────────────────────────────────────────
videoUpload?.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/x-ms-wmv', 'video/x-m4v'];
  if (!allowedTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv|wmv|m4v)$/i)) {
    setSystemMessage('Unsupported video type. Please upload MP4, AVI, MOV, MKV, or WMV.', true);
    videoUpload.value = '';
    return;
  }

  const MAX_VIDEO_MB = 500;
  if (file.size > MAX_VIDEO_MB * 1024 * 1024) {
    setSystemMessage(`Video too large. Maximum allowed size is ${MAX_VIDEO_MB} MB.`, true);
    videoUpload.value = '';
    return;
  }

  const formData = new FormData();
  formData.append('video', file);

  try {
    setSystemMessage(`Uploading video "${file.name}"… this may take a moment.`);

    const response = await fetch('/upload_video', { method: 'POST', body: formData });

    if (response.status === 413) {
      setSystemMessage('Video rejected by server: file too large.', true);
      return;
    }

    const result = await response.json();
    if (!result.ok) {
      setSystemMessage(result.message, true);
      return;
    }

    // Switch display back to the stream (detection runs on the video now)
    previewImage.style.display = 'none';
    streamImage.style.display  = 'block';
    streamImage.src            = `/video_feed?ts=${Date.now()}`;
    setSystemMessage(`${result.message} — Video will loop automatically.`);
    startPolling();
    fetchStats();
  } catch {
    setSystemMessage('Video upload failed. Is the backend running?', true);
  } finally {
    videoUpload.value = '';
  }
});
startPolling();
fetchStats();