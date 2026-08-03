// Real WebSocket + mic wiring for the live-call screen. History and settings screens are
// still visual-only placeholders (REWRITE_PLAN.md §9 describes them, but no history-persistence
// or settings-config API has been built yet) — see the TODO markers below.

const SERVER_SAMPLE_RATE = 16000;
const MAX_CHART_POINTS = 40;

let audioContext = null;
let workletNode = null;
let mediaStream = null;
let ws = null;
let callStartTime = null;
let durationTimer = null;
let riskHistory = []; // {score, level}

// ---- theme toggle ----
const themeBtns = document.querySelectorAll('#themeToggle button');
themeBtns.forEach((btn) =>
  btn.addEventListener('click', () => {
    themeBtns.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    const t = btn.dataset.theme;
    if (t === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', t);
  })
);

// ---- tabs ----
document.querySelectorAll('.tab').forEach((tab) =>
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');
    document.querySelectorAll('.screen').forEach((s) => s.classList.remove('active'));
    document.getElementById('screen-' + tab.dataset.screen).classList.add('active');
  })
);

// ---- chart ----
function buildPath(pts, w, h) {
  if (pts.length === 0) return { line: `M0,${h} L${w},${h}`, area: `M0,${h} L${w},${h} Z`, endX: w, endY: h };
  const step = pts.length > 1 ? w / (pts.length - 1) : 0;
  const toY = (v) => h - (v / 100) * h;
  let line = `M0,${toY(pts[0])}`;
  pts.forEach((v, i) => { if (i > 0) line += ` L${i * step},${toY(v)}`; });
  return { line, area: line + ` L${w},${h} L0,${h} Z`, endX: pts.length > 1 ? w : 0, endY: toY(pts[pts.length - 1]) };
}

function renderChart() {
  const scores = riskHistory.map((r) => r.score);
  const isAlertColor = riskHistory.length > 0 && riskHistory[riskHistory.length - 1].level === 'high';
  const { line, area, endX, endY } = buildPath(scores, 400, 92);

  document.getElementById('chartArea').setAttribute('d', area);
  document.getElementById('chartLine').setAttribute('d', line);
  document.getElementById('chartEndpoint').setAttribute('cx', endX);
  document.getElementById('chartEndpoint').setAttribute('cy', endY);

  ['chartArea', 'chartLine', 'chartEndpoint'].forEach((id) =>
    document.getElementById(id).classList.toggle('alert', isAlertColor)
  );

  const latest = riskHistory[riskHistory.length - 1];
  document.getElementById('riskNum').textContent = latest ? latest.score : 0;
  const pill = document.getElementById('riskPill');
  const level = latest ? latest.level : 'low';
  const label = { low: '低風險', medium: '中風險', high: '高風險' }[level];
  pill.className = 'risk-pill ' + level;
  pill.textContent = label;
}

function pushRiskPoint(score, level) {
  riskHistory.push({ score, level });
  if (riskHistory.length > MAX_CHART_POINTS) riskHistory.shift();
  renderChart();
}

// ---- transcript ----
function appendTranscriptRow(msg, flagged) {
  const list = document.getElementById('transcriptList');
  const placeholder = document.getElementById('transcriptPlaceholder');
  if (placeholder) placeholder.remove();

  const row = document.createElement('div');
  row.className = 't-row' + (flagged ? ' flagged' : '');
  const mm = String(Math.floor(msg.timestamp / 60)).padStart(2, '0');
  const ss = String(Math.floor(msg.timestamp % 60)).padStart(2, '0');
  row.innerHTML = `
    <div class="t-time mono">${mm}:${ss}</div>
    <div class="t-content">${msg.transcript_text}</div>
  `;
  list.appendChild(row);
  row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ---- alert ----
function showAlert(alertMsg) {
  const banner = document.getElementById('alertBanner');
  document.querySelector('#alertBanner .alert-title').textContent =
    alertMsg.reason === 'hard_trigger' ? `高信度示警 · ${alertMsg.trigger_name || ''}` : '示警 · 風險持續偏高';
  document.querySelector('#alertBanner .alert-reason').textContent = alertMsg.justification || '';
  banner.classList.add('show');

  const rows = document.querySelectorAll('#transcriptList .t-row');
  if (rows.length > 0) rows[rows.length - 1].classList.add('flagged');
}

document.querySelector('.btn-ack').addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'ack_alert' }));
  document.getElementById('alertBanner').classList.remove('show');
});
document.querySelector('.btn-dismiss').addEventListener('click', () => {
  document.getElementById('alertBanner').classList.remove('show');
});

// ---- server messages ----
function handleServerMessage(event) {
  const msg = JSON.parse(event.data);
  if (msg.type === 'chunk_update') {
    appendTranscriptRow(msg, msg.risk_level === 'high');
    pushRiskPoint(msg.chunk_risk_score, msg.risk_level);
  } else if (msg.type === 'alert') {
    showAlert(msg);
  }
}

// ---- call duration ----
function updateDuration() {
  const secs = Math.floor((Date.now() - callStartTime) / 1000);
  const m = String(Math.floor(secs / 60)).padStart(2, '0');
  const s = String(secs % 60).padStart(2, '0');
  document.getElementById('callDuration').textContent = `${m}:${s}`;
}

// ---- listening state ----
function setListeningUI(isListening) {
  const statusLabel = document.querySelector('.call-meta-label');
  const stopBtn = document.getElementById('btnToggleListen');
  document.querySelector('.pulse-dot').style.opacity = isListening ? '1' : '0.25';
  statusLabel.textContent = isListening ? '監聽中 · 來電' : '尚未開始監聽';
  stopBtn.textContent = isListening ? '結束監聽' : '開始監聽';
  stopBtn.classList.toggle('btn-stop', isListening);
}

async function startListening() {
  audioContext = new AudioContext({ sampleRate: SERVER_SAMPLE_RATE });
  if (audioContext.sampleRate !== SERVER_SAMPLE_RATE) {
    // See audio-worklet.js comment — this browser didn't honor the requested rate.
    console.warn(
      `AudioContext sampleRate is ${audioContext.sampleRate}, not ${SERVER_SAMPLE_RATE}. ` +
        'The backend VAD chunker assumes 16kHz frames — resampling in the worklet is needed ' +
        'before this will work correctly on this browser.'
    );
  }

  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
  await audioContext.audioWorklet.addModule('audio-worklet.js');
  const source = audioContext.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioContext, 'pcm-capture-processor');

  ws = new WebSocket(`ws://${location.host}/ws/call`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = handleServerMessage;
  ws.onerror = (e) => console.error('WebSocket error', e);
  ws.onopen = () => {
    workletNode.port.onmessage = (event) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(event.data.buffer);
    };
    source.connect(workletNode);
  };

  callStartTime = Date.now();
  durationTimer = setInterval(updateDuration, 1000);
  riskHistory = [];
  renderChart();
  setListeningUI(true);
}

function stopListening() {
  if (ws) {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'stop' }));
    ws.close();
    ws = null;
  }
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  if (audioContext) audioContext.close();
  clearInterval(durationTimer);
  setListeningUI(false);
}

document.getElementById('btnToggleListen').addEventListener('click', () => {
  if (ws) stopListening();
  else startListening().catch((err) => {
    console.error('Failed to start listening', err);
    alert('無法啟動麥克風監聽：' + err.message);
  });
});

// ---- settings: visual-only for now (TODO: wire to a real config API) ----
const debounceSlider = document.getElementById('debounceSlider');
const debounceVal = document.getElementById('debounceVal');
debounceSlider.addEventListener('input', () => {
  debounceVal.textContent = debounceSlider.value + ' 句';
});
document.querySelectorAll('.effort-opt').forEach((opt) =>
  opt.addEventListener('click', () => {
    document.querySelectorAll('.effort-opt').forEach((o) => o.classList.remove('active'));
    opt.classList.add('active');
  })
);

renderChart();
