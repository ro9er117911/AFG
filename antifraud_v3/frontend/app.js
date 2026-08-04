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
let latestMicLevel = 0;
let micLevelRafId = null;

// ---- mic level meter ----
// Computed straight from the same Float32 frames we're about to ship over the WebSocket, so
// this reflects exactly what the server is receiving (not a separate/possibly-desynced tap).
function computeAudioLevel(samples) {
  let sumSquares = 0;
  for (let i = 0; i < samples.length; i++) sumSquares += samples[i] * samples[i];
  const rms = Math.sqrt(sumSquares / samples.length);
  return Math.min(1, rms * 6); // speech RMS is typically well under 1.0 — scale up so the meter is readable
}

function tickMicLevel() {
  document.getElementById('micLevelFill').style.width = `${Math.round(latestMicLevel * 100)}%`;
  micLevelRafId = requestAnimationFrame(tickMicLevel);
}

function stopMicLevel() {
  if (micLevelRafId) cancelAnimationFrame(micLevelRafId);
  micLevelRafId = null;
  latestMicLevel = 0;
  document.getElementById('micLevelFill').style.width = '0%';
}

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
    if (tab.dataset.screen === 'testdata') loadTestData();
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

// ---- pitch (acoustic) chart ----
// Separate from the risk chart above: this streams live during the call itself (ASR/acoustic/
// emotion, no LLM — pipeline/chunk_worker.py's process_chunk_signals), one point per chunk.
// The risk chart only ever gets one point, once the end-of-call LLM analysis finishes.
const PITCH_MIN_HZ = 60;
const PITCH_MAX_HZ = 320;
const PITCH_CHART_H = 56;

function buildPitchPath(pts, w, h) {
  if (pts.length === 0) return { line: `M0,${h} L${w},${h}`, area: `M0,${h} L${w},${h} Z`, endX: w, endY: h };
  const step = pts.length > 1 ? w / (pts.length - 1) : 0;
  const toY = (hz) => {
    const clamped = Math.min(PITCH_MAX_HZ, Math.max(PITCH_MIN_HZ, hz));
    return h - ((clamped - PITCH_MIN_HZ) / (PITCH_MAX_HZ - PITCH_MIN_HZ)) * h;
  };
  let line = `M0,${toY(pts[0])}`;
  pts.forEach((v, i) => { if (i > 0) line += ` L${i * step},${toY(v)}`; });
  return { line, area: line + ` L${w},${h} L0,${h} Z`, endX: pts.length > 1 ? w : 0, endY: toY(pts[pts.length - 1]) };
}

// els takes actual DOM elements (not ids) so this also works scoped inside one Test Data card
// via card.querySelector(), where there can be several cards' worth of charts on the page at
// once and a global id lookup wouldn't disambiguate them.
function makePitchChartRenderer(els) {
  let points = [];
  function render() {
    const { line, area, endX, endY } = buildPitchPath(points, 400, PITCH_CHART_H);
    els.area.setAttribute('d', area);
    els.line.setAttribute('d', line);
    els.endpoint.setAttribute('cx', endX);
    els.endpoint.setAttribute('cy', endY);
  }
  return {
    push(hz) {
      if (hz == null) return; // no voiced segment in this chunk — see chunk_worker.py's _compact_acoustic
      points.push(hz);
      if (points.length > MAX_CHART_POINTS) points.shift();
      render();
    },
    reset() {
      points = [];
      render();
    },
  };
}

function pitchChartEls(prefix) {
  return {
    area: document.getElementById(prefix + 'PitchArea'),
    line: document.getElementById(prefix + 'PitchLine'),
    endpoint: document.getElementById(prefix + 'PitchEndpoint'),
  };
}

// ---- emotion badge + acoustic readout ----
const EMOTION_LABELS_ZH = { anger: '生氣', boredom: '無聊', disgust: '厭惡', fear: '恐懼', happy: '開心', neutral: '中性', sad: '難過' };

function emotionBadgeClass(label) {
  if (label === 'happy') return 'emo-low';
  if (label === 'fear' || label === 'sad') return 'emo-medium';
  if (label === 'anger' || label === 'disgust') return 'emo-high';
  return 'emo-neutral'; // neutral, boredom
}

function formatAcousticReadout(a) {
  const pitch = a.mean_pitch != null ? `${a.mean_pitch.toFixed(0)} Hz` : '無足夠有聲段';
  return `音高 ${pitch} · jitter ${a.jitter_local}% · shimmer ${a.shimmer_local}% · HNR ${a.hnr}dB · 停頓 ${a.pause_ratio}%`;
}

// ---- "what did the AI see" transparency ----
// evidence is exactly the ChunkEvidence sent to reasoning/discriminate.py's prompt (see
// pipeline/chunk_worker.py's run_final_analysis) — shown verbatim, not summarized further, so
// there's no gap between what the user sees here and what the model actually received.
function evidenceHTML(evidence) {
  const esc = (s) => (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return `
    <details class="evidence-details">
      <summary>查看送給 AI 的內容</summary>
      <div class="evidence-block"><span class="label">逐字稿</span>${esc(evidence.transcript_segment)}<span class="label">聲學摘要</span>${esc(evidence.acoustic_summary)}<span class="label">情緒摘要</span>${esc(evidence.emotion_summary)}</div>
    </details>`;
}

// ---- generic streaming-call view controller ----
// Shared by the live screen, the upload-result screen, and each Test Data card — all three now
// consume the same event vocabulary (chunk_update/alert/final_analysis, see server/ws.py and
// server/upload.py's stream_pipeline_over_audio), just wired to different DOM elements. els:
// { transcriptList, pitchChart, acousticReadout, emotionBadge, alertsList?, finalSummary?,
//   evidence? } — the optional ones don't exist on the live screen (which uses the top alert
// banner and the settled riskPill/riskNum readout instead).
function createStreamingView(els) {
  let alerts = [];
  return {
    reset() {
      alerts = [];
      els.transcriptList.innerHTML = '';
      els.pitchChart.reset();
      els.acousticReadout.textContent = '尚未偵測到聲音。';
      els.emotionBadge.textContent = '尚無資料';
      els.emotionBadge.className = 'emo-badge emo-neutral';
      if (els.alertsList) els.alertsList.innerHTML = '';
      if (els.finalSummary) els.finalSummary.textContent = '';
      if (els.evidence) els.evidence.innerHTML = '';
    },
    onChunk(msg) {
      const row = appendTranscriptRowInto(els.transcriptList, msg, false);
      if (els.autoScroll) row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      if (msg.acoustic) {
        els.pitchChart.push(msg.acoustic.mean_pitch);
        els.acousticReadout.textContent = formatAcousticReadout(msg.acoustic);
      }
      if (msg.emotion) {
        els.emotionBadge.textContent = `${EMOTION_LABELS_ZH[msg.emotion.label] || msg.emotion.label} ${msg.emotion.top_prob}`;
        els.emotionBadge.className = 'emo-badge ' + emotionBadgeClass(msg.emotion.label);
      }
    },
    onAlert(msg) {
      alerts.push(msg);
      if (els.alertsList) els.alertsList.innerHTML = alertsListHTML(alerts);
      const rows = els.transcriptList.querySelectorAll('.t-row');
      if (rows.length > 0) rows[rows.length - 1].classList.add('flagged');
    },
    onFinal(msg) {
      if (els.finalSummary) {
        els.finalSummary.textContent = `${RISK_LABELS[msg.risk_level] || msg.risk_level} — ${msg.justification || ''}`;
      }
      if (els.evidence && msg.evidence) els.evidence.innerHTML = evidenceHTML(msg.evidence);
    },
    getAlerts() {
      return alerts;
    },
  };
}

// Reads a fetch() Response whose body is newline-delimited JSON (server/upload.py,
// server/testdata.py) and calls onEvent(parsedObject) for each complete line as it arrives —
// this is what makes the upload/test-data screens progressive instead of waiting for the
// whole response before showing anything.
async function readNdjsonStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (value) buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (line) onEvent(JSON.parse(line));
    }
    if (done) break;
  }
  const tail = buf.trim();
  if (tail) onEvent(JSON.parse(tail));
}

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

// ---- interim (still-speaking) transcript preview ----
// server/ws.py's _interim_transcript_loop re-transcribes the in-progress utterance every
// ~1.8s and pushes it here so text shows up while the caller is still talking, instead of only
// once the full turn + risk analysis finishes. This is a provisional ASR-only guess (no
// acoustic/emotion/LLM pass yet) — kept as one live row that updates in place, not appended
// to history, and removed the moment the real, fully-analyzed chunk_update for that utterance
// arrives (see handleServerMessage).
let interimRow = null;

function setInterimTranscript(text) {
  const listEl = document.getElementById('transcriptList');
  if (!text) {
    if (interimRow) {
      interimRow.remove();
      interimRow = null;
    }
    return;
  }
  if (!interimRow) {
    const placeholder = listEl.querySelector('.t-placeholder');
    if (placeholder) placeholder.remove();
    interimRow = document.createElement('div');
    interimRow.className = 't-row t-interim';
    interimRow.innerHTML = `<div class="t-time mono">…</div><div class="t-content"></div>`;
    listEl.appendChild(interimRow);
  }
  interimRow.querySelector('.t-content').textContent = text;
  interimRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Shared by every screen that shows fired alerts (live banner aside — see showAlert): the
// upload-result screen, the history detail view, and the Test Data screen.
function alertsListHTML(alerts) {
  if (alerts.length === 0) return '<div class="placeholder-note">分析過程中沒有觸發任何示警。</div>';
  return alerts
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

// Renders a full call record — { transcript, risk_trajectory, alerts, final_risk_level,
// final_justification, duration_seconds } (the shape server/upload.py's response and GET
// /api/calls/{id} both return) — into a chart + alerts list + transcript list. Shared by the
// upload-result screen and the history detail view so the two don't duplicate this markup.
// ctx: { chart, listEl, alertsEl, summaryEl, emptyMessage, summaryText(data) }.
//
// risk_trajectory now holds at most one point — the single end-of-call analysis (see
// pipeline/chunk_worker.py's run_final_analysis) — not one point per transcript line, so it's
// no longer paired with transcript by index. A transcript line is flagged instead if a fired
// alert's quoted phrase appears in it, which still highlights the specific line that mattered.
function renderCallDetail(data, ctx) {
  ctx.chart.reset();
  ctx.listEl.innerHTML = '';

  const flagQuotes = data.alerts.map((a) => a.quote).filter(Boolean);
  data.transcript.forEach((t) => {
    const flagged = flagQuotes.some((q) => t.text.includes(q));
    appendTranscriptRowInto(ctx.listEl, { timestamp: t.timestamp, transcript_text: t.text }, flagged);
  });
  if (data.transcript.length === 0) {
    ctx.listEl.innerHTML = `<div class="t-placeholder">${ctx.emptyMessage}</div>`;
  }
  data.risk_trajectory.forEach((rp) => ctx.chart.push(rp.chunk_risk_score, rp.risk_level));

  ctx.alertsEl.innerHTML = alertsListHTML(data.alerts);
  ctx.summaryEl.textContent = ctx.summaryText(data);
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

// Wires the live-call screen's transcript/pitch-chart/acoustic-readout/emotion-badge to the
// shared streaming view controller (see createStreamingView above). finalSummary/evidence
// point at the "最終研判" panel's text + collapsible; there's no alertsList here — the live
// screen uses the top alert-banner (showAlert) instead of a list panel.
const liveView = createStreamingView({
  transcriptList: document.getElementById('transcriptList'),
  pitchChart: makePitchChartRenderer(pitchChartEls('live')),
  acousticReadout: document.getElementById('liveAcousticReadout'),
  emotionBadge: document.getElementById('liveEmotionBadge'),
  finalSummary: document.getElementById('finalJustification'),
  evidence: document.getElementById('liveEvidence'),
  autoScroll: true,
});

// ---- server messages ----
function handleServerMessage(event) {
  const msg = JSON.parse(event.data);
  if (msg.type === 'chunk_update') {
    // No risk_level here anymore — acoustic/ASR/emotion run live per chunk, but the LLM
    // judgment only runs once at call end (see "final_analysis" below). A row is only
    // flagged live if the cheap keyword hard-trigger check fires for it (a separate "alert"
    // message, handled by showAlert below).
    hideSystemMessage();
    setInterimTranscript(''); // the real transcript row below replaces the live preview
    liveView.onChunk(msg);
  } else if (msg.type === 'interim_transcript') {
    setInterimTranscript(msg.text);
  } else if (msg.type === 'final_analysis') {
    // The one LLM analysis for the whole call, pushed once after "結束監聽" — see
    // pipeline/chunk_worker.py's run_final_analysis.
    liveChart.push(msg.chunk_risk_score, msg.risk_level);
    liveView.onFinal(msg);
  } else if (msg.type === 'alert') {
    showAlert(msg); // the top banner + flags the last transcript row
  } else if (msg.type === 'error') {
    // Surfaces e.g. a missing/invalid ANTHROPIC_API_KEY, or an unavailable `claude` CLI (see
    // .env.example) — confirmed by testing that without this, the pipeline error silently
    // dropped the connection instead.
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
  ws.onopen = () => {
    hideSystemMessage();
    workletNode.port.onmessage = (event) => {
      const samples = event.data;
      latestMicLevel = latestMicLevel * 0.7 + computeAudioLevel(samples) * 0.3;
      if (ws.readyState === WebSocket.OPEN) ws.send(samples.buffer);
    };
    source.connect(workletNode);
  };
  ws.onclose = (e) => {
    if (!e.wasClean) showSystemMessage('與伺服器的連線已中斷。');
    ws = null;
    // A stuck "still speaking" preview with no risk score would otherwise linger forever if
    // the connection ends without one final chunk_update to clear it (see setInterimTranscript).
    setInterimTranscript('');
    // Only reset the status label if the user hasn't already started a new call in the
    // meantime (that call's own audioContext would be set, and owns the label now).
    if (!audioContext) document.querySelector('.call-meta-label').textContent = '尚未開始監聽';
  };

  callStartTime = Date.now();
  durationTimer = setInterval(updateDuration, 1000);
  liveChart.reset();
  liveView.reset();
  document.getElementById('transcriptList').innerHTML =
    '<div class="t-placeholder">正在等待語音輸入…</div>';
  document.getElementById('finalJustification').textContent =
    '聲學／語音辨識／情緒分析在通話中即時顯示於下方；完整研判會在「結束監聽」後才跑一次，結果顯示在這裡。';
  setInterimTranscript('');
  setListeningUI(true);
  micLevelRafId = requestAnimationFrame(tickMicLevel);
}

function stopListening() {
  // Stop capturing first, but do NOT close the socket here: the server still needs it open to
  // finish analyzing the last utterance (ASR + acoustic + LLM calls take real time) and send
  // back its chunk_update/alert before it closes the connection on its own. Closing from the
  // client right after sending "stop" was racing that final result off the wire — see the
  // matching comment in server/ws.py's call_socket() finally-block.
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  if (audioContext) audioContext.close();
  mediaStream = null;
  audioContext = null;
  stopMicLevel();
  clearInterval(durationTimer);
  setListeningUI(false);

  if (ws && ws.readyState === WebSocket.OPEN) {
    document.querySelector('.call-meta-label').textContent = '正在處理最後結果…';
    ws.send(JSON.stringify({ type: 'stop' }));
  }
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
  // Navigating to the History tab always lands on the list, even if a detail view was left
  // open from a previous visit.
  document.getElementById('historyDetailView').style.display = 'none';
  document.getElementById('historyListView').style.display = 'block';

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
    <div class="history-row" data-call-id="${c.id}">
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

// Delegated once on the container (not inside loadHistory, which replaces its innerHTML on
// every tab visit / refresh — attaching there would stack up a duplicate listener per visit).
document.getElementById('historyList').addEventListener('click', (e) => {
  const row = e.target.closest('.history-row');
  if (row) openHistoryDetail(row.dataset.callId);
});

const historyChart = makeChartRenderer({
  area: 'historyChartArea', line: 'historyChartLine', endpoint: 'historyChartEndpoint',
  riskNum: 'historyRiskNum', riskPill: 'historyRiskPill',
});

async function openHistoryDetail(callId) {
  document.getElementById('historyListView').style.display = 'none';
  document.getElementById('historyDetailView').style.display = 'block';
  document.getElementById('historyDetailSummary').textContent = '載入中…';
  document.getElementById('historyDetailAlertsList').innerHTML = '';
  document.getElementById('historyDetailTranscriptList').innerHTML = '';

  let data;
  try {
    const res = await fetch(`/api/calls/${callId}`);
    if (!res.ok) throw new Error(`伺服器回應 ${res.status}`);
    data = await res.json();
  } catch (err) {
    document.getElementById('historyDetailSummary').textContent = '無法載入這通電話的紀錄：' + err.message;
    return;
  }

  renderCallDetail(data, {
    chart: historyChart,
    listEl: document.getElementById('historyDetailTranscriptList'),
    alertsEl: document.getElementById('historyDetailAlertsList'),
    summaryEl: document.getElementById('historyDetailSummary'),
    emptyMessage: '這通電話沒有偵測到語音內容。',
    summaryText: (d) =>
      `${formatWhen(d.started_at)} · ${d.source === 'upload' ? '上傳錄音分析' : '通話'} #${d.id} · ` +
      `整體風險：${RISK_LABELS[d.final_risk_level] || '低風險'} · 長度 ${formatDuration(d.duration_seconds)} · 偵測到 ${d.transcript.length} 句` +
      (d.risk_trajectory[0]?.justification ? ` · ${d.risk_trajectory[0].justification}` : ''),
  });
}

document.getElementById('btnHistoryBack').addEventListener('click', () => {
  document.getElementById('historyDetailView').style.display = 'none';
  document.getElementById('historyListView').style.display = 'block';
});

// ---- settings (server/api.py: GET/POST /api/settings) ----
const providerSelect = document.getElementById('providerSelect');
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

  providerSelect.value = settings.llm_provider;
  modelSelect.value = settings.llm_model;
  effortRow.querySelectorAll('.effort-opt').forEach((opt) =>
    opt.classList.toggle('active', opt.dataset.value === settings.llm_effort)
  );
  currentHardTriggers = [...settings.hard_triggers];
  renderHardTriggers();

  // Also reflect the active model/effort in the live-call header chip.
  document.querySelector('.model-chip').textContent = `${settings.llm_model} · ${settings.llm_effort}`;
}

providerSelect.addEventListener('change', () => saveSettings({ llm_provider: providerSelect.value }));
modelSelect.addEventListener('change', () => saveSettings({ llm_model: modelSelect.value }));
effortRow.querySelectorAll('.effort-opt').forEach((opt) =>
  opt.addEventListener('click', () => {
    effortRow.querySelectorAll('.effort-opt').forEach((o) => o.classList.remove('active'));
    opt.classList.add('active');
    saveSettings({ llm_effort: opt.dataset.value });
  })
);

// ---- upload analysis (server/upload.py: POST /api/upload-call, streamed NDJSON) ----
const uploadView = createStreamingView({
  transcriptList: document.getElementById('uploadTranscriptList'),
  pitchChart: makePitchChartRenderer(pitchChartEls('upload')),
  acousticReadout: document.getElementById('uploadAcousticReadout'),
  emotionBadge: document.getElementById('uploadEmotionBadge'),
  alertsList: document.getElementById('uploadAlertsList'),
  finalSummary: document.getElementById('uploadSummary'),
  evidence: document.getElementById('uploadEvidence'),
});

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
  uploadView.reset();
  uploadChart.reset();
  document.getElementById('uploadResult').style.display = 'block'; // shown immediately — chunk_update events fill it in progressively

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/api/upload-call', { method: 'POST', body: form });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `伺服器回應 ${res.status}`);
    }

    let doneMsg = null;
    let finalMsg = null;
    await readNdjsonStream(res, (msg) => {
      if (msg.type === 'chunk_update') uploadView.onChunk(msg);
      else if (msg.type === 'alert') uploadView.onAlert(msg);
      else if (msg.type === 'final_analysis') {
        uploadView.onFinal(msg);
        uploadChart.push(msg.chunk_risk_score, msg.risk_level);
        finalMsg = msg;
      } else if (msg.type === 'error') throw new Error(msg.message);
      else if (msg.type === 'done') doneMsg = msg;
    });

    if (doneMsg) {
      document.getElementById('uploadSummary').textContent =
        `整體風險：${RISK_LABELS[doneMsg.final_risk_level] || '低風險'} · 音訊長度 ${formatDuration(doneMsg.duration_seconds)} · ` +
        `已存入歷史紀錄 #${doneMsg.call_id}` + (finalMsg ? ` · ${finalMsg.justification}` : '');
      loadHistory(); // the uploaded call now shows up in history too — keep it in sync if visible
    }
  } catch (err) {
    console.error('Upload analysis failed', err);
    errEl.textContent = '分析失敗：' + err.message;
    errEl.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
}

document.getElementById('btnUploadAnalyze').addEventListener('click', analyzeUpload);

// ---- test data (server/testdata.py: GET /api/test-clips, POST /api/test-clips/.../analyze) ----
// Replays eval/test_clips/ audio through the exact same pipeline as a live call/upload, for
// spot-checking transcript + risk judgment quality without polluting call history — results
// here are never persisted (server/testdata.py's docstring), so revisiting the tab re-fetches
// the clip list with a clean, unrun state each time.
const TEST_CLIP_COUNT = 3;

function testClipCardHTML(category, filename) {
  return `
  <div class="panel test-clip-card" data-category="${category}" data-filename="${filename}">
    <div class="panel-head">
      <div class="panel-title">${filename}</div>
      <button class="btn-toggle-listen test-clip-run">執行分析</button>
    </div>
    <audio controls src="/api/test-clips/${category}/${filename}/audio"></audio>
    <div class="test-clip-result" style="display:none;">
      <div class="field-help mono test-clip-summary"></div>
      <div class="panel-head" style="margin-top:10px; margin-bottom:6px;">
        <div class="panel-title" style="font-size:11px;">聲學／情緒</div>
        <span class="emo-badge emo-neutral test-clip-emotion-badge">尚無資料</span>
      </div>
      <div class="pitch-chart-wrap">
        <svg viewBox="0 0 400 56" preserveAspectRatio="none">
          <path class="pitch-area test-clip-pitch-area" d="M0,56 L400,56 Z"/>
          <path class="pitch-line test-clip-pitch-line" d="M0,56 L400,56"/>
          <circle class="chart-endpoint test-clip-pitch-endpoint" style="fill:var(--accent)" cx="0" cy="56" r="3"/>
        </svg>
      </div>
      <div class="field-help mono test-clip-acoustic-readout">尚未偵測到聲音。</div>
      <div class="field-help mono test-clip-final-summary" style="margin-top:6px;"></div>
      <div class="test-clip-evidence"></div>
      <div class="test-clip-alerts" style="margin-top:10px;"></div>
      <div class="transcript-list test-clip-transcript" style="margin-top:10px;"></div>
    </div>
    <div class="placeholder-note test-clip-placeholder">尚未執行分析。</div>
  </div>`;
}

// Builds a createStreamingView bound to one card's scoped elements (not global ids — several
// cards can be on screen at once, see pitchChartEls's docstring for why makePitchChartRenderer
// takes elements directly).
function createTestClipView(card) {
  return createStreamingView({
    transcriptList: card.querySelector('.test-clip-transcript'),
    pitchChart: makePitchChartRenderer({
      area: card.querySelector('.test-clip-pitch-area'),
      line: card.querySelector('.test-clip-pitch-line'),
      endpoint: card.querySelector('.test-clip-pitch-endpoint'),
    }),
    acousticReadout: card.querySelector('.test-clip-acoustic-readout'),
    emotionBadge: card.querySelector('.test-clip-emotion-badge'),
    alertsList: card.querySelector('.test-clip-alerts'),
    finalSummary: card.querySelector('.test-clip-final-summary'),
    evidence: card.querySelector('.test-clip-evidence'),
  });
}

async function loadTestData() {
  const benignEl = document.getElementById('testDataBenign');
  const scamEl = document.getElementById('testDataScam');
  let clips;
  try {
    clips = await (await fetch('/api/test-clips')).json();
  } catch (err) {
    benignEl.innerHTML = `<div class="placeholder-note">無法載入測試音檔清單：${err.message}</div>`;
    scamEl.innerHTML = '';
    return;
  }
  benignEl.innerHTML = clips.benign
    .slice(0, TEST_CLIP_COUNT)
    .map((f) => testClipCardHTML('benign', f))
    .join('');
  scamEl.innerHTML = clips.scam
    .slice(0, TEST_CLIP_COUNT)
    .map((f) => testClipCardHTML('scam', f))
    .join('');
}

async function runTestClip(card) {
  const category = card.dataset.category;
  const filename = card.dataset.filename;
  const btn = card.querySelector('.test-clip-run');
  const resultEl = card.querySelector('.test-clip-result');
  const placeholderEl = card.querySelector('.test-clip-placeholder');

  btn.disabled = true;
  const originalLabel = btn.textContent;
  btn.textContent = '分析中…';
  placeholderEl.textContent = '分析中…（逐句語音辨識／聲學/情緒會先顯示，LLM 最終研判最後才出現）';
  placeholderEl.style.display = 'block';

  const view = createTestClipView(card);
  view.reset();
  resultEl.style.display = 'block'; // shown immediately — chunk_update events fill it in progressively

  try {
    const res = await fetch(`/api/test-clips/${category}/${filename}/analyze`, { method: 'POST' });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `伺服器回應 ${res.status}`);
    }

    let doneMsg = null;
    await readNdjsonStream(res, (msg) => {
      if (msg.type === 'chunk_update') view.onChunk(msg);
      else if (msg.type === 'alert') view.onAlert(msg);
      else if (msg.type === 'final_analysis') view.onFinal(msg);
      else if (msg.type === 'error') throw new Error(msg.message);
      else if (msg.type === 'done') doneMsg = msg;
    });

    if (doneMsg) {
      card.querySelector('.test-clip-summary').textContent =
        `整體風險：${RISK_LABELS[doneMsg.final_risk_level] || '低風險'} · 音訊長度 ${formatDuration(doneMsg.duration_seconds)} · ` +
        `分析耗時 ${doneMsg.processing_seconds.toFixed(1)} 秒`;
    }
    placeholderEl.style.display = 'none';
  } catch (err) {
    console.error('Test clip analysis failed', err);
    placeholderEl.textContent = '分析失敗：' + err.message;
    placeholderEl.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
}

document.getElementById('screen-testdata').addEventListener('click', (e) => {
  if (e.target.classList.contains('test-clip-run')) runTestClip(e.target.closest('.test-clip-card'));
});

liveChart.render();
loadSettings();
