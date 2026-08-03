// Real WebSocket + mic wiring for the live-call screen, plus REST calls for the History and
// Settings screens (server/api.py) — see docs/DESIGN.md §9.

const SERVER_SAMPLE_RATE = 16000;
const MAX_CHART_POINTS = 40;

let audioContext = null;
let workletNode = null;
let mediaStream = null;
let ws = null;
let callStartTime = null;
let durationTimer = null;

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
    if (tab.dataset.screen === 'history') loadHistory();
    if (tab.dataset.screen === 'settings') loadSettings();
  })
);

// ---- chart ----
// Shared by both the live screen and the upload-result screen (see docs task: "reuse the
// existing risk-trajectory chart / transcript-list rendering code rather than duplicating
// it"). makeChartRenderer() closes over one screen's DOM elements and its own risk-history
// array, so the two screens don't stomp on each other's state when switching tabs.
function buildPath(pts, w, h) {
  if (pts.length === 0) return { line: `M0,${h} L${w},${h}`, area: `M0,${h} L${w},${h} Z`, endX: w, endY: h };
  const step = pts.length > 1 ? w / (pts.length - 1) : 0;
  const toY = (v) => h - (v / 100) * h;
  let line = `M0,${toY(pts[0])}`;
  pts.forEach((v, i) => { if (i > 0) line += ` L${i * step},${toY(v)}`; });
  return { line, area: line + ` L${w},${h} L0,${h} Z`, endX: pts.length > 1 ? w : 0, endY: toY(pts[pts.length - 1]) };
}

function makeChartRenderer(ids) {
  const els = {
    area: document.getElementById(ids.area),
    line: document.getElementById(ids.line),
    endpoint: document.getElementById(ids.endpoint),
    riskNum: document.getElementById(ids.riskNum),
    riskPill: document.getElementById(ids.riskPill),
  };
  let riskHistory = []; // {score, level}

  function render() {
    const scores = riskHistory.map((r) => r.score);
    const isAlertColor = riskHistory.length > 0 && riskHistory[riskHistory.length - 1].level === 'high';
    const { line, area, endX, endY } = buildPath(scores, 400, 92);

    els.area.setAttribute('d', area);
    els.line.setAttribute('d', line);
    els.endpoint.setAttribute('cx', endX);
    els.endpoint.setAttribute('cy', endY);
    [els.area, els.line, els.endpoint].forEach((el) => el.classList.toggle('alert', isAlertColor));

    const latest = riskHistory[riskHistory.length - 1];
    els.riskNum.textContent = latest ? latest.score : 0;
    const level = latest ? latest.level : 'low';
    els.riskPill.className = 'risk-pill ' + level;
    els.riskPill.textContent = { low: '低風險', medium: '中風險', high: '高風險' }[level];
  }

  return {
    push(score, level) {
      riskHistory.push({ score, level });
      if (riskHistory.length > MAX_CHART_POINTS) riskHistory.shift();
      render();
    },
    reset() {
      riskHistory = [];
      render();
    },
    render,
  };
}

const liveChart = makeChartRenderer({
  area: 'chartArea', line: 'chartLine', endpoint: 'chartEndpoint', riskNum: 'riskNum', riskPill: 'riskPill',
});
const uploadChart = makeChartRenderer({
  area: 'uploadChartArea', line: 'uploadChartLine', endpoint: 'uploadChartEndpoint',
  riskNum: 'uploadRiskNum', riskPill: 'uploadRiskPill',
});

// ---- transcript ----
// Generic row-append usable by both screens; the live path additionally auto-scrolls, so it
// keeps a thin wrapper below.
function appendTranscriptRowInto(listEl, msg, flagged) {
  const placeholder = listEl.querySelector('.t-placeholder');
  if (placeholder) placeholder.remove();

  const row = document.createElement('div');
  row.className = 't-row' + (flagged ? ' flagged' : '');
  const mm = String(Math.floor(msg.timestamp / 60)).padStart(2, '0');
  const ss = String(Math.floor(msg.timestamp % 60)).padStart(2, '0');
  row.innerHTML = `
    <div class="t-time mono">${mm}:${ss}</div>
    <div class="t-content">${msg.transcript_text}</div>
  `;
  listEl.appendChild(row);
  return row;
}

function appendTranscriptRow(msg, flagged) {
  const row = appendTranscriptRowInto(document.getElementById('transcriptList'), msg, flagged);
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

// ---- system status (connection/pipeline errors — kept visually distinct from the fraud
// alert banner above, see index.html's .system-banner comment) ----
function showSystemMessage(text) {
  const banner = document.getElementById('systemBanner');
  banner.textContent = text;
  banner.classList.add('show');
}
function hideSystemMessage() {
  document.getElementById('systemBanner').classList.remove('show');
}

// ---- server messages ----
function handleServerMessage(event) {
  const msg = JSON.parse(event.data);
  if (msg.type === 'chunk_update') {
    hideSystemMessage();
    appendTranscriptRow(msg, msg.risk_level === 'high');
    liveChart.push(msg.chunk_risk_score, msg.risk_level);
  } else if (msg.type === 'alert') {
    showAlert(msg);
  } else if (msg.type === 'error') {
    // Surfaces e.g. a missing/invalid ANTHROPIC_API_KEY (see .env.example) — confirmed by
    // testing that without this, the pipeline error silently dropped the connection instead.
    showSystemMessage('分析發生問題：' + msg.message);
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
  ws.onerror = (e) => {
    console.error('WebSocket error', e);
    showSystemMessage('連線發生錯誤，請檢查伺服器是否正常運作。');
  };
  ws.onclose = (e) => {
    if (!e.wasClean) showSystemMessage('與伺服器的連線已中斷。');
  };
  ws.onopen = () => {
    hideSystemMessage();
    workletNode.port.onmessage = (event) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(event.data.buffer);
    };
    source.connect(workletNode);
  };

  callStartTime = Date.now();
  durationTimer = setInterval(updateDuration, 1000);
  liveChart.reset();
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

// ---- history (server/api.py: GET /api/calls) ----
const RISK_LABELS = { low: '低風險', medium: '中風險', high: '高風險' };

function formatWhen(unixSeconds) {
  const d = new Date(unixSeconds * 1000);
  return d.toLocaleString('zh-TW', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
function formatDuration(seconds) {
  const s = Math.round(seconds || 0);
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

async function loadHistory() {
  const container = document.getElementById('historyList');
  let calls;
  try {
    calls = await (await fetch('/api/calls')).json();
  } catch (err) {
    container.innerHTML = `<div class="placeholder-note">無法載入歷史紀錄：${err.message}</div>`;
    return;
  }
  if (!calls.length) {
    container.innerHTML = '<div class="placeholder-note" id="historyPlaceholder">還沒有通話紀錄——結束一次「開始監聽」就會出現在這裡。</div>';
    return;
  }
  container.innerHTML = calls
    .map(
      (c) => `
    <div class="history-row">
      <div>
        <div class="history-when">${formatWhen(c.started_at)}</div>
        <div class="history-caller">${c.source === 'upload' ? '上傳錄音分析' : '通話'} #${c.id}${c.ended_reason === 'disconnected' ? '（連線中斷結束）' : ''}</div>
      </div>
      <div class="history-dur mono">${formatDuration(c.duration_seconds)}</div>
      <span class="risk-pill ${c.final_risk_level || 'low'}">${RISK_LABELS[c.final_risk_level] || '低風險'}</span>
    </div>`
    )
    .join('');
}

// ---- settings (server/api.py: GET/POST /api/settings) ----
const debounceSlider = document.getElementById('debounceSlider');
const debounceVal = document.getElementById('debounceVal');
const modelSelect = document.getElementById('modelSelect');
const effortRow = document.getElementById('effortRow');
const hardTriggerList = document.getElementById('hardTriggerList');

let currentHardTriggers = [];

function renderHardTriggers() {
  hardTriggerList.innerHTML =
    currentHardTriggers.map((t, i) => `<span class="chip">${t} <button data-idx="${i}">&times;</button></span>`).join('') +
    '<button class="chip-add" id="chipAddBtn">+ 新增</button>';

  hardTriggerList.querySelectorAll('.chip button').forEach((btn) =>
    btn.addEventListener('click', () => {
      currentHardTriggers.splice(Number(btn.dataset.idx), 1);
      renderHardTriggers();
      saveSettings({ hard_triggers: currentHardTriggers });
    })
  );
  document.getElementById('chipAddBtn').addEventListener('click', () => {
    const text = prompt('新增立即示警關鍵字（描述句，不是精確比對字串）：');
    if (text && text.trim()) {
      currentHardTriggers.push(text.trim());
      renderHardTriggers();
      saveSettings({ hard_triggers: currentHardTriggers });
    }
  });
}

async function saveSettings(partial) {
  try {
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(partial),
    });
  } catch (err) {
    console.error('Failed to save settings', err);
  }
}

async function loadSettings() {
  let settings;
  try {
    settings = await (await fetch('/api/settings')).json();
  } catch (err) {
    console.error('Failed to load settings', err);
    return;
  }

  modelSelect.value = settings.llm_model;
  debounceSlider.value = settings.debounce_chunks;
  debounceVal.textContent = settings.debounce_chunks + ' 句';
  effortRow.querySelectorAll('.effort-opt').forEach((opt) =>
    opt.classList.toggle('active', opt.dataset.value === settings.llm_effort)
  );
  currentHardTriggers = [...settings.hard_triggers];
  renderHardTriggers();

  // Also reflect the active model/effort in the live-call header chip.
  document.querySelector('.model-chip').textContent = `${settings.llm_model} · ${settings.llm_effort}`;
}

modelSelect.addEventListener('change', () => saveSettings({ llm_model: modelSelect.value }));
debounceSlider.addEventListener('input', () => {
  debounceVal.textContent = debounceSlider.value + ' 句';
});
debounceSlider.addEventListener('change', () => saveSettings({ debounce_chunks: Number(debounceSlider.value) }));
effortRow.querySelectorAll('.effort-opt').forEach((opt) =>
  opt.addEventListener('click', () => {
    effortRow.querySelectorAll('.effort-opt').forEach((o) => o.classList.remove('active'));
    opt.classList.add('active');
    saveSettings({ llm_effort: opt.dataset.value });
  })
);

// ---- upload analysis (server/upload.py: POST /api/upload-call) ----
async function analyzeUpload() {
  const fileInput = document.getElementById('uploadFileInput');
  const file = fileInput.files[0];
  const errEl = document.getElementById('uploadError');
  errEl.style.display = 'none';
  if (!file) {
    errEl.textContent = '請先選擇音訊檔案。';
    errEl.style.display = 'block';
    return;
  }

  const btn = document.getElementById('btnUploadAnalyze');
  btn.disabled = true;
  const originalLabel = btn.textContent;
  btn.textContent = '分析中…';
  document.getElementById('uploadResult').style.display = 'none';

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/api/upload-call', { method: 'POST', body: form });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `伺服器回應 ${res.status}`);
    }
    const data = await res.json();
    renderUploadResult(data);
    loadHistory(); // the uploaded call now shows up in history too — keep it in sync if visible
  } catch (err) {
    console.error('Upload analysis failed', err);
    errEl.textContent = '分析失敗：' + err.message;
    errEl.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
}

function renderUploadResult(data) {
  uploadChart.reset();
  const listEl = document.getElementById('uploadTranscriptList');
  listEl.innerHTML = '';

  data.transcript.forEach((t, i) => {
    const rp = data.risk_trajectory[i];
    const flagged = !!rp && rp.risk_level === 'high';
    appendTranscriptRowInto(listEl, { timestamp: t.timestamp, transcript_text: t.text }, flagged);
    if (rp) uploadChart.push(rp.chunk_risk_score, rp.risk_level);
  });
  if (data.transcript.length === 0) {
    listEl.innerHTML = '<div class="t-placeholder">沒有偵測到語音內容，可能是錄音太短或音量太小。</div>';
  }

  const alertsEl = document.getElementById('uploadAlertsList');
  if (data.alerts.length === 0) {
    alertsEl.innerHTML = '<div class="placeholder-note">分析過程中沒有觸發任何示警。</div>';
  } else {
    alertsEl.innerHTML = data.alerts
      .map(
        (a) => `
      <div class="alert-banner show">
        <div class="alert-stripe"></div>
        <div class="alert-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 3L2 20h20L12 3z" stroke="#fff" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v4M12 17h.01" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>
        </div>
        <div class="alert-body">
          <div class="alert-title">${a.reason === 'hard_trigger' ? '高信度示警 · ' + (a.trigger_name || '') : '示警 · 風險持續偏高'}</div>
          <div class="alert-reason">${a.justification || ''}</div>
        </div>
      </div>`
      )
      .join('');
  }

  document.getElementById('uploadSummary').textContent =
    `整體風險：${RISK_LABELS[data.final_risk_level] || '低風險'} · 音訊長度 ${formatDuration(data.duration_seconds)} · 偵測到 ${data.transcript.length} 句 · 已存入歷史紀錄 #${data.call_id}`;

  document.getElementById('uploadResult').style.display = 'block';
}

document.getElementById('btnUploadAnalyze').addEventListener('click', analyzeUpload);

liveChart.render();
loadSettings();
