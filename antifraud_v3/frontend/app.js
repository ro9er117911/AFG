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

// ---- acoustic metric charts (pitch + 3a's jitter/shimmer/HNR/pause/speech-rate) ----
// Separate from the risk chart above: these stream live during the call itself (ASR/acoustic/
// emotion, no LLM — pipeline/chunk_worker.py's process_chunk_signals), one point per chunk.
// The risk chart only ever gets one point, once the end-of-call LLM analysis finishes.
//
// buildMetricPath/makeMetricChartRenderer generalize what used to be pitch-only rendering code
// (buildPitchPath/makePitchChartRenderer) so the same machinery draws all six acoustic
// sparklines, each with its own sensible domain — see METRIC_DEFS below.
const PITCH_CHART_H = 56;
const METRIC_CHART_H = 34;

const METRIC_DEFS = {
  mean_pitch: { label: '音高', unit: 'Hz', min: 60, max: 320, digits: 0, h: PITCH_CHART_H },
  jitter_local: { label: 'Jitter', unit: '%', min: 0, max: 5, digits: 2, h: METRIC_CHART_H },
  shimmer_local: { label: 'Shimmer', unit: '%', min: 0, max: 10, digits: 2, h: METRIC_CHART_H },
  hnr: { label: 'HNR', unit: 'dB', min: 0, max: 30, digits: 1, h: METRIC_CHART_H },
  pause_ratio: { label: '停頓佔比', unit: '%', min: 0, max: 90, digits: 1, h: METRIC_CHART_H },
  speech_rate_variation: { label: '語速變化', unit: '', min: 0, max: 10, digits: 2, h: METRIC_CHART_H },
};
// 3a's small-multiples grid — pitch keeps its own larger chart in the existing panel, these five
// get the compact grid (see index.html's .metric-grid).
const GRID_METRIC_KEYS = ['jitter_local', 'shimmer_local', 'hnr', 'pause_ratio', 'speech_rate_variation'];

function buildMetricPath(pts, w, h, min, max) {
  const toY = (v) => {
    const clamped = Math.min(max, Math.max(min, v));
    return h - ((clamped - min) / (max - min)) * h;
  };
  if (pts.length === 0) return { line: `M0,${h} L${w},${h}`, area: `M0,${h} L${w},${h} Z`, endX: w, endY: h, toY };
  const step = pts.length > 1 ? w / (pts.length - 1) : 0;
  let line = `M0,${toY(pts[0])}`;
  pts.forEach((v, i) => { if (i > 0) line += ` L${i * step},${toY(v)}`; });
  return { line, area: line + ` L${w},${h} L0,${h} Z`, endX: pts.length > 1 ? w : 0, endY: toY(pts[pts.length - 1]), toY };
}

// els takes actual DOM elements (not ids) so this also works scoped inside one Test Data card
// via card.querySelector(), where there can be several cards' worth of charts on the page at
// once and a global id lookup wouldn't disambiguate them. els.endpoint/els.baselineLine are
// optional (the compact 3a grid charts skip the endpoint dot to stay visually quiet).
function makeMetricChartRenderer(els, { min, max, h }) {
  let points = [];
  let baselineVal = null;
  function render() {
    const { line, area, endX, endY, toY } = buildMetricPath(points, 400, h, min, max);
    els.area.setAttribute('d', area);
    els.line.setAttribute('d', line);
    if (els.endpoint) {
      els.endpoint.setAttribute('cx', endX);
      els.endpoint.setAttribute('cy', endY);
    }
    if (els.baselineLine) {
      els.baselineLine.setAttribute('d', baselineVal == null ? '' : `M0,${toY(baselineVal)} L400,${toY(baselineVal)}`);
    }
  }
  return {
    push(v) {
      if (v == null) return; // e.g. no voiced segment in this chunk — see chunk_worker.py's _compact_acoustic
      points.push(v);
      if (points.length > MAX_CHART_POINTS) points.shift();
      render();
    },
    setBaseline(v) {
      baselineVal = v == null ? null : v;
      render();
    },
    reset() {
      points = [];
      baselineVal = null;
      render();
    },
    values() {
      return points;
    },
  };
}

function pitchChartEls(prefix) {
  return {
    area: document.getElementById(prefix + 'PitchArea'),
    line: document.getElementById(prefix + 'PitchLine'),
    endpoint: document.getElementById(prefix + 'PitchEndpoint'),
    baselineLine: document.getElementById(prefix + 'PitchBaseline'),
  };
}

// camelCase DOM-id suffix for a snake_case metric key, e.g. "pause_ratio" -> "PauseRatio".
function metricIdSuffix(key) {
  return key.split('_').map((s) => s.charAt(0).toUpperCase() + s.slice(1)).join('');
}

function metricChartEls(prefix, key) {
  const suf = metricIdSuffix(key);
  return {
    area: document.getElementById(prefix + suf + 'Area'),
    line: document.getElementById(prefix + suf + 'Line'),
    baselineLine: document.getElementById(prefix + suf + 'Baseline'),
  };
}

function metricChartElsScoped(card, key) {
  const kebab = key.replace(/_/g, '-');
  return {
    area: card.querySelector(`.test-clip-${kebab}-area`),
    line: card.querySelector(`.test-clip-${kebab}-line`),
    baselineLine: card.querySelector(`.test-clip-${kebab}-baseline`),
  };
}

// Builds one renderer per GRID_METRIC_KEYS entry (+ pitch, handled separately by the caller
// since it lives in a different-sized chart) — shared shape used by createStreamingView.
function makeGridChartRenderers(elsForKey) {
  const renderers = {};
  GRID_METRIC_KEYS.forEach((key) => {
    const def = METRIC_DEFS[key];
    renderers[key] = makeMetricChartRenderer(elsForKey(key), { min: def.min, max: def.max, h: def.h });
  });
  return renderers;
}

// ---- 3b: baseline readout ----
function formatBaselineReadout(baseline) {
  if (!baseline) return '尚未建立基準值（通話開頭 15 秒後自動建立）。';
  return (
    `此通話基準：音高 ${baseline.mean_pitch.toFixed(0)}Hz · jitter ${baseline.jitter_local.toFixed(2)}% · ` +
    `shimmer ${baseline.shimmer_local.toFixed(2)}% · HNR ${baseline.hnr.toFixed(1)}dB · ` +
    `停頓 ${baseline.pause_ratio.toFixed(1)}% · 語速變化 ${baseline.speech_rate_variation.toFixed(2)}`
  );
}

function formatBaselineDelta(current, baseline) {
  if (!baseline || current.mean_pitch == null || !baseline.mean_pitch) return '';
  const pct = ((current.mean_pitch - baseline.mean_pitch) / baseline.mean_pitch) * 100;
  const sign = pct >= 0 ? '+' : '';
  return ` · 音高較基準 ${sign}${pct.toFixed(0)}%`;
}

// ---- 3c: 8-dim acoustic "deception profile" radar — visualization only, see index.html's
// .radar-disclaimer text. Each axis maps 1:1 to a single already-streamed acoustic metric (no
// combining multiple indicators into one axis — that's the shape of antifraud_v2's
// normalization bug, see docs/DESIGN.md §1). Computed entirely client-side from data already
// pushed to this view; never touches pipeline/reasoning code, so it cannot leak into risk
// scoring the way the old system did.
const RADAR_AXES = [
  { key: 'pitch_instability', invert: false },
  { key: 'shimmer_local', invert: false },
  { key: 'mean_pitch', invert: false },
  { key: 'hnr', invert: true }, // lower HNR = more of this indicator
  { key: 'speech_rate_variation', invert: false },
  { key: 'pause_ratio', invert: false },
  { key: 'mean_volume', invert: false },
  { key: 'jitter_local', invert: false },
];

// z = (current - this call's own opening baseline) / (std dev of this metric's values seen so
// far in this call), clamped to [-3,+3] and rescaled to [0,100] — 50 means "same as this call's
// baseline". Self-referential to THIS call's own data only, not v2's external fixed thresholds.
function zScoreTo100(current, baselineVal, values, invert) {
  if (current == null || baselineVal == null || values.length < 2) return 50;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  const std = Math.max(Math.sqrt(variance), 1e-6);
  let z = (current - baselineVal) / std;
  if (invert) z = -z;
  z = Math.max(-3, Math.min(3, z));
  return ((z + 3) / 6) * 100;
}

function polarPoint(cx, cy, r, angleDeg) {
  const rad = (Math.PI / 180) * angleDeg;
  return [cx + r * Math.sin(rad), cy - r * Math.cos(rad)];
}

function buildRadarPath(values8, cx, cy, maxR) {
  const n = values8.length;
  const pts = values8.map((v, i) => polarPoint(cx, cy, (Math.max(0, Math.min(100, v)) / 100) * maxR, i * (360 / n)));
  return 'M' + pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' L') + ' Z';
}

// history: plain object { metricKey: number[] }, accumulated by createStreamingView.onChunk.
function makeRadarChartRenderer(shapeEl) {
  function render(history, baseline) {
    const values8 = RADAR_AXES.map((axis) => {
      const vals = history[axis.key] || [];
      const current = vals.length > 0 ? vals[vals.length - 1] : null;
      const baselineVal = baseline ? baseline[axis.key] : null;
      return zScoreTo100(current, baselineVal, vals, axis.invert);
    });
    shapeEl.setAttribute('d', buildRadarPath(values8, 110, 118, 85));
  }
  return {
    render,
    reset() {
      shapeEl.setAttribute('d', buildRadarPath([50, 50, 50, 50, 50, 50, 50, 50], 110, 118, 85));
    },
  };
}

// ---- HTML generator for the 3a/3b/3c panel shared by live/upload/test-clip screens ----
// classIdAttr(prefix, useClass, fixedClasses, name) returns a single class="..." attribute
// (plus id="..." when not useClass) for a bare camelCase element name (e.g. "radarShape",
// "jitterLocalArea") combined with the element's own fixed CSS class(es) — see classIdAttr below.
// Returns a single `class="..."` attribute (id-based mode also appends a separate `id="..."`)
// — MUST stay a single class attribute per element: two class="..." attributes on one tag is
// invalid HTML, and browsers silently keep only the first, dropping the second (confirmed by a
// real headless-browser run: this exact bug made every scoped .test-clip-* selector return
// null, which then threw inside makeMetricChartRenderer's els.area.setAttribute(...)).
function classIdAttr(prefix, useClass, fixedClasses, name) {
  if (useClass) {
    const kebab = name.replace(/([A-Z])/g, '-$1').toLowerCase();
    return `class="${fixedClasses} test-clip-${kebab}"`;
  }
  const id = `${prefix}${name.charAt(0).toUpperCase()}${name.slice(1)}`;
  return `class="${fixedClasses}" id="${id}"`;
}

function acousticExtraHTML(prefix, useClass) {
  const a = (fixedClasses, name) => classIdAttr(prefix, useClass, fixedClasses, name);
  const metricCells = GRID_METRIC_KEYS.map((key) => {
    const def = METRIC_DEFS[key];
    const suf = metricIdSuffix(key);
    const base = suf.charAt(0).toLowerCase() + suf.slice(1);
    return `
    <div class="metric-cell">
      <div class="metric-cell-label">${def.label}</div>
      <div class="metric-chart-wrap">
        <svg viewBox="0 0 400 ${METRIC_CHART_H}" preserveAspectRatio="none">
          <path ${a('metric-area', base + 'Area')} d="M0,${METRIC_CHART_H} L400,${METRIC_CHART_H} Z"/>
          <path ${a('metric-line', base + 'Line')} d="M0,${METRIC_CHART_H} L400,${METRIC_CHART_H}"/>
          <path ${a('baseline-ref', base + 'Baseline')} d=""/>
        </svg>
      </div>
    </div>`;
  }).join('');

  return `
    <div class="metric-grid">${metricCells}</div>
    <div ${a('baseline-readout mono', 'baselineReadout')}>尚未建立基準值（通話開頭 15 秒後自動建立）。</div>
    <div class="radar-wrap">
      <svg viewBox="-20 -2 260 240">
        <circle class="radar-grid-ring" cx="110" cy="118" r="21.25"/>
        <circle class="radar-grid-ring" cx="110" cy="118" r="42.5"/>
        <circle class="radar-grid-ring" cx="110" cy="118" r="63.75"/>
        <circle class="radar-grid-ring" cx="110" cy="118" r="85"/>
        <line class="radar-axis-line" x1="110" y1="118" x2="110" y2="33"/>
        <line class="radar-axis-line" x1="110" y1="118" x2="170.1" y2="57.9"/>
        <line class="radar-axis-line" x1="110" y1="118" x2="195" y2="118"/>
        <line class="radar-axis-line" x1="110" y1="118" x2="170.1" y2="178.1"/>
        <line class="radar-axis-line" x1="110" y1="118" x2="110" y2="203"/>
        <line class="radar-axis-line" x1="110" y1="118" x2="49.9" y2="178.1"/>
        <line class="radar-axis-line" x1="110" y1="118" x2="25" y2="118"/>
        <line class="radar-axis-line" x1="110" y1="118" x2="49.9" y2="57.9"/>
        <text class="radar-axis-label" x="110" y="18">攻擊性語氣</text>
        <text class="radar-axis-label" x="180.7" y="47.3">矛盾衝突</text>
        <text class="radar-axis-label" x="212" y="121">明確否認</text>
        <text class="radar-axis-label" x="180.7" y="192">尷尬掩蓋</text>
        <text class="radar-axis-label" x="110" y="230">警覺避談</text>
        <text class="radar-axis-label" x="39.3" y="192">猶豫不決</text>
        <text class="radar-axis-label" x="8" y="121">異常興奮</text>
        <text class="radar-axis-label" x="39.3" y="47.3">邏輯漏洞</text>
        <path ${a('radar-shape', 'radarShape')} d="M110,118 L110,118 L110,118 L110,118 L110,118 L110,118 L110,118 L110,118 Z"/>
      </svg>
      <div class="radar-disclaimer">聲學特徵剖面（僅供參考，非風險判定）— 與本通話自己的開頭基準值比較</div>
    </div>`;
}

// ---- emotion badge + acoustic readout ----
const EMOTION_LABELS_ZH = { anger: '生氣', boredom: '無聊', disgust: '厭惡', fear: '恐懼', happy: '開心', neutral: '中性', sad: '難過' };
const FRAUD_TYPE_LABELS_ZH = {
  investment_fraud: '投資詐騙', phishing_fraud: '網路釣魚詐騙', identity_theft: '身分冒用',
  lottery_fraud: '中獎摸彩詐騙', banking_fraud: '銀行詐騙', extortion_fraud: '勒索詐騙',
  customer_service_fraud: '客服詐騙', unclassified: '未分類',
};

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

function badgeRowHTML(msg) {
  const chips = [];
  if (msg.fraud_type && msg.fraud_type.fraud_type && msg.fraud_type.fraud_type !== 'unclassified') {
    const label = FRAUD_TYPE_LABELS_ZH[msg.fraud_type.fraud_type] || msg.fraud_type.fraud_type;
    chips.push(`<span class="info-badge">疑似：${label}（${msg.fraud_type.confidence}）</span>`);
  }
  if (msg.audio_quality && msg.audio_quality.narrowband) {
    chips.push(`<span class="info-badge narrowband">電話頻寬（窄頻）— 聲學細節可信度較低</span>`);
  }
  return chips.join('');
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
// { transcriptList, pitchChart, gridCharts, acousticReadout, emotionBadge, baselineReadout?,
//   radar?, badgeRow?, alertsList?, finalSummary?, evidence? } — the optional ones don't exist
// on the live screen (which uses the top alert banner and the settled riskPill/riskNum readout
// instead).
function createStreamingView(els) {
  let alerts = [];
  let baseline = null;
  const metricHistory = { mean_pitch: [], pitch_instability: [], mean_volume: [] }; // radar-only trackers
  return {
    reset() {
      alerts = [];
      baseline = null;
      Object.keys(metricHistory).forEach((k) => { metricHistory[k] = []; });
      els.transcriptList.innerHTML = '';
      els.pitchChart.reset();
      if (els.gridCharts) Object.values(els.gridCharts).forEach((c) => c.reset());
      if (els.baselineReadout) els.baselineReadout.textContent = formatBaselineReadout(null);
      if (els.radar) els.radar.reset();
      if (els.badgeRow) els.badgeRow.innerHTML = '';
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
        const a = msg.acoustic;
        els.pitchChart.push(a.mean_pitch);
        if (els.gridCharts) GRID_METRIC_KEYS.forEach((key) => els.gridCharts[key].push(a[key]));
        els.acousticReadout.textContent = formatAcousticReadout(a) + formatBaselineDelta(a, baseline);
        if (a.mean_pitch != null) metricHistory.mean_pitch.push(a.mean_pitch);
        if (a.pitch_instability != null) metricHistory.pitch_instability.push(a.pitch_instability);
        metricHistory.mean_volume.push(a.mean_volume);
        if (els.radar) {
          const fullHistory = {
            ...metricHistory,
            jitter_local: els.gridCharts ? els.gridCharts.jitter_local.values() : [],
            shimmer_local: els.gridCharts ? els.gridCharts.shimmer_local.values() : [],
            hnr: els.gridCharts ? els.gridCharts.hnr.values() : [],
            pause_ratio: els.gridCharts ? els.gridCharts.pause_ratio.values() : [],
            speech_rate_variation: els.gridCharts ? els.gridCharts.speech_rate_variation.values() : [],
          };
          els.radar.render(fullHistory, baseline);
        }
      }
      if (msg.baseline) {
        baseline = msg.baseline;
        els.pitchChart.setBaseline(baseline.mean_pitch);
        if (els.gridCharts) GRID_METRIC_KEYS.forEach((key) => els.gridCharts[key].setBaseline(baseline[key]));
        if (els.baselineReadout) els.baselineReadout.textContent = formatBaselineReadout(baseline);
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
      if (els.badgeRow) els.badgeRow.innerHTML = badgeRowHTML(msg);
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
document.getElementById('liveAcousticExtra').innerHTML = acousticExtraHTML('live', false);
document.getElementById('uploadAcousticExtra').innerHTML = acousticExtraHTML('upload', false);

const liveView = createStreamingView({
  transcriptList: document.getElementById('transcriptList'),
  pitchChart: makeMetricChartRenderer(pitchChartEls('live'), { min: METRIC_DEFS.mean_pitch.min, max: METRIC_DEFS.mean_pitch.max, h: PITCH_CHART_H }),
  gridCharts: makeGridChartRenderers((key) => metricChartEls('live', key)),
  acousticReadout: document.getElementById('liveAcousticReadout'),
  emotionBadge: document.getElementById('liveEmotionBadge'),
  baselineReadout: document.getElementById('liveBaselineReadout'),
  radar: makeRadarChartRenderer(document.getElementById('liveRadarShape')),
  badgeRow: document.getElementById('liveBadgeRow'),
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

// ---- 3d: waveform + spectrogram (upload-analysis and test-data screens only; static,
// one-shot render after the whole file is decoded — not real-time/streaming, per the plan's
// priority call). Self-contained (no new library): a small iterative radix-2 Cooley-Tukey FFT
// for the spectrogram, matching this project's existing "no new framework" convention.
async function decodeAudioFile(arrayBuffer) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  try {
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
    return { samples: audioBuffer.getChannelData(0), sampleRate: audioBuffer.sampleRate };
  } finally {
    ctx.close();
  }
}

function drawWaveform(canvas, samples) {
  const w = (canvas.width = Math.max(1, Math.floor(canvas.clientWidth || 400)));
  const h = (canvas.height = 80);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  if (samples.length === 0) return;
  const mid = h / 2;
  const samplesPerPixel = Math.max(1, Math.floor(samples.length / w));
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#2F8F86';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < w; x++) {
    const start = x * samplesPerPixel;
    if (start >= samples.length) break;
    const end = Math.min(samples.length, start + samplesPerPixel);
    let min = 1;
    let max = -1;
    for (let i = start; i < end; i++) {
      const v = samples[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x + 0.5, mid + min * mid);
    ctx.lineTo(x + 0.5, mid + max * mid);
  }
  ctx.stroke();
}

// In-place iterative radix-2 Cooley-Tukey — `real`/`imag` length must be a power of 2.
function fftInPlace(real, imag) {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr0 = Math.cos(ang);
    const wi0 = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curWr = 1;
      let curWi = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = real[i + k];
        const ui = imag[i + k];
        const vr = real[i + k + len / 2] * curWr - imag[i + k + len / 2] * curWi;
        const vi = real[i + k + len / 2] * curWi + imag[i + k + len / 2] * curWr;
        real[i + k] = ur + vr;
        imag[i + k] = ui + vi;
        real[i + k + len / 2] = ur - vr;
        imag[i + k + len / 2] = ui - vi;
        const nwr = curWr * wr0 - curWi * wi0;
        const nwi = curWr * wi0 + curWi * wr0;
        curWr = nwr;
        curWi = nwi;
      }
    }
  }
}

const SPECTROGRAM_FFT_SIZE = 1024;
const SPECTROGRAM_HOP = 256;

function computeSpectrogram(samples) {
  const n = SPECTROGRAM_FFT_SIZE;
  const window = new Float32Array(n);
  for (let i = 0; i < n; i++) window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)); // Hann
  const frames = [];
  for (let start = 0; start + n <= samples.length; start += SPECTROGRAM_HOP) {
    const real = new Float32Array(n);
    const imag = new Float32Array(n);
    for (let i = 0; i < n; i++) real[i] = samples[start + i] * window[i];
    fftInPlace(real, imag);
    const half = n / 2;
    const mags = new Float32Array(half);
    for (let i = 0; i < half; i++) mags[i] = 20 * Math.log10(Math.sqrt(real[i] * real[i] + imag[i] * imag[i]) + 1e-6);
    frames.push(mags);
  }
  return frames;
}

// Three-stop gradient (dark -> teal accent -> warm high-magnitude) — a fixed palette rather
// than reading CSS custom properties per-pixel, since canvas pixel colors don't need to react
// to a live theme toggle for a static, one-shot render.
function magnitudeToColorRGB(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const stops = [
    [10, 20, 19],
    [47, 143, 134],
    [226, 86, 76],
  ];
  const seg = clamped < 0.5 ? 0 : 1;
  const localT = clamped < 0.5 ? clamped / 0.5 : (clamped - 0.5) / 0.5;
  const a = stops[seg];
  const b = stops[seg + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * localT),
    Math.round(a[1] + (b[1] - a[1]) * localT),
    Math.round(a[2] + (b[2] - a[2]) * localT),
  ];
}

function drawSpectrogram(canvas, frames) {
  const w = (canvas.width = Math.max(1, Math.floor(canvas.clientWidth || 400)));
  const h = (canvas.height = 110);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  if (frames.length === 0) return;
  const numBins = frames[0].length;
  let minDb = Infinity;
  let maxDb = -Infinity;
  for (const f of frames) {
    for (const v of f) {
      if (v < minDb) minDb = v;
      if (v > maxDb) maxDb = v;
    }
  }
  const range = Math.max(maxDb - minDb, 1e-6);
  const img = ctx.createImageData(w, h);
  for (let x = 0; x < w; x++) {
    const frameIdx = Math.min(frames.length - 1, Math.floor((x / w) * frames.length));
    const frame = frames[frameIdx];
    for (let y = 0; y < h; y++) {
      const binIdx = Math.min(numBins - 1, Math.floor(((h - 1 - y) / h) * numBins));
      const t = (frame[binIdx] - minDb) / range;
      const [r, g, b] = magnitudeToColorRGB(t);
      const idx = (y * w + x) * 4;
      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

async function renderWaveformAndSpectrogram(arrayBuffer, waveformCanvas, spectrogramCanvas) {
  try {
    const { samples } = await decodeAudioFile(arrayBuffer);
    drawWaveform(waveformCanvas, samples);
    drawSpectrogram(spectrogramCanvas, computeSpectrogram(samples));
  } catch (err) {
    console.error('Failed to render waveform/spectrogram', err);
  }
}

// ---- upload analysis (server/upload.py: POST /api/upload-call, streamed NDJSON) ----
const uploadView = createStreamingView({
  transcriptList: document.getElementById('uploadTranscriptList'),
  pitchChart: makeMetricChartRenderer(pitchChartEls('upload'), { min: METRIC_DEFS.mean_pitch.min, max: METRIC_DEFS.mean_pitch.max, h: PITCH_CHART_H }),
  gridCharts: makeGridChartRenderers((key) => metricChartEls('upload', key)),
  acousticReadout: document.getElementById('uploadAcousticReadout'),
  emotionBadge: document.getElementById('uploadEmotionBadge'),
  baselineReadout: document.getElementById('uploadBaselineReadout'),
  radar: makeRadarChartRenderer(document.getElementById('uploadRadarShape')),
  badgeRow: document.getElementById('uploadBadgeRow'),
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

  file.arrayBuffer().then((buf) =>
    renderWaveformAndSpectrogram(buf, document.getElementById('uploadWaveform'), document.getElementById('uploadSpectrogram'))
  );

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
    <canvas class="waveform-canvas test-clip-waveform"></canvas>
    <canvas class="spectrogram-canvas test-clip-spectrogram"></canvas>
    <div class="test-clip-result" style="display:none;">
      <div class="field-help mono test-clip-summary"></div>
      <div class="badge-row test-clip-badge-row"></div>
      <div class="panel-head" style="margin-top:10px; margin-bottom:6px;">
        <div class="panel-title" style="font-size:11px;">聲學／情緒</div>
        <span class="emo-badge emo-neutral test-clip-emotion-badge">尚無資料</span>
      </div>
      <div class="pitch-chart-wrap">
        <svg viewBox="0 0 400 56" preserveAspectRatio="none">
          <path class="pitch-area test-clip-pitch-area" d="M0,56 L400,56 Z"/>
          <path class="pitch-line test-clip-pitch-line" d="M0,56 L400,56"/>
          <path class="baseline-ref test-clip-pitch-baseline" d=""/>
          <circle class="chart-endpoint test-clip-pitch-endpoint" style="fill:var(--accent)" cx="0" cy="56" r="3"/>
        </svg>
      </div>
      <div class="field-help mono test-clip-acoustic-readout">尚未偵測到聲音。</div>
      ${acousticExtraHTML('', true)}
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
    pitchChart: makeMetricChartRenderer(
      {
        area: card.querySelector('.test-clip-pitch-area'),
        line: card.querySelector('.test-clip-pitch-line'),
        endpoint: card.querySelector('.test-clip-pitch-endpoint'),
        baselineLine: card.querySelector('.test-clip-pitch-baseline'),
      },
      { min: METRIC_DEFS.mean_pitch.min, max: METRIC_DEFS.mean_pitch.max, h: PITCH_CHART_H }
    ),
    gridCharts: makeGridChartRenderers((key) => metricChartElsScoped(card, key)),
    acousticReadout: card.querySelector('.test-clip-acoustic-readout'),
    emotionBadge: card.querySelector('.test-clip-emotion-badge'),
    baselineReadout: card.querySelector('.test-clip-baseline-readout'),
    radar: makeRadarChartRenderer(card.querySelector('.test-clip-radar-shape')),
    badgeRow: card.querySelector('.test-clip-badge-row'),
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

  fetch(`/api/test-clips/${category}/${filename}/audio`)
    .then((r) => r.arrayBuffer())
    .then((buf) =>
      renderWaveformAndSpectrogram(buf, card.querySelector('.test-clip-waveform'), card.querySelector('.test-clip-spectrogram'))
    )
    .catch((err) => console.error('Failed to load test clip audio for waveform/spectrogram', err));

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
