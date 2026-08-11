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
    // ECharts bakes literal color values into each chart's option — a theme flip needs every
    // live instance to re-read the new CSS var values and re-render, or they'd stay stuck on
    // the old theme's colors until their next data update. See CHART_INSTANCES below.
    CHART_INSTANCES.forEach((c) => c.render());
  })
);
// "auto" theme also needs to react to the OS-level scheme changing while the tab is open, not
// just to the buttons above.
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (!document.documentElement.hasAttribute('data-theme')) CHART_INSTANCES.forEach((c) => c.render());
});

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
    // A chart initialized while its screen was display:none (zero width) needs an explicit
    // resize once the screen becomes visible again — ECharts doesn't observe this on its own.
    CHART_INSTANCES.forEach((c) => c.resize());
  })
);

window.addEventListener('resize', () => CHART_INSTANCES.forEach((c) => c.resize()));

// ---- shared ECharts plumbing ----
// One global list of { render, resize } so the theme toggle can recolor every live chart
// instance (ECharts bakes in literal color values, not CSS custom properties, so a theme flip
// has to explicitly re-read the CSS vars and call setOption again) and so switching tabs can
// force a remeasure (a chart initialized while its screen was display:none reports zero width
// until told to resize).
const CHART_INSTANCES = [];
function registerChart(entry) {
  CHART_INSTANCES.push(entry);
  return entry;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function riskColor(level) {
  return { low: cssVar('--risk-low'), medium: cssVar('--risk-medium'), high: cssVar('--risk-high') }[level] || cssVar('--risk-low');
}

// Reuses an already-initialized instance on the same container instead of calling echarts.init()
// twice — test-clip cards rebuild their chart-renderer closures on every "執行分析" click
// (createTestClipView), but the underlying <div> is the same DOM node across re-runs.
function ensureEChart(containerEl) {
  return echarts.getInstanceByDom(containerEl) || echarts.init(containerEl);
}

// ---- risk gauge ----
// Replaces the old single-point "trend line": the pipeline only ever produces one risk score
// per call now (reasoning/fusion.py's build_synthesize_result runs once, at call end), so a
// line/area chart never had more than one point to draw — it was always just a flat shape. A
// gauge represents what the data actually is: one 0-100 reading with a color band.
function makeRiskGauge(containerEl, pillEl) {
  let score = 0;
  let level = 'low';

  function render() {
    const chart = ensureEChart(containerEl);
    chart.setOption(
      {
        series: [
          {
            type: 'gauge',
            startAngle: 210,
            endAngle: -30,
            min: 0,
            max: 100,
            radius: '100%',
            axisLine: { lineStyle: { width: 12, color: [[1, cssVar('--surface-2')]] } },
            progress: { show: true, width: 12, itemStyle: { color: riskColor(level) } },
            axisTick: { show: false },
            splitLine: { show: false },
            axisLabel: { show: false },
            pointer: { show: false },
            anchor: { show: false },
            detail: {
              valueAnimation: true,
              fontSize: 28,
              fontWeight: 700,
              color: cssVar('--ink'),
              offsetCenter: [0, '-8%'],
              formatter: '{value}',
            },
            data: [{ value: score }],
          },
        ],
      },
      true
    );
    if (pillEl) {
      pillEl.className = 'risk-pill ' + level;
      pillEl.textContent = { low: '低風險', medium: '中風險', high: '高風險' }[level];
    }
  }

  registerChart({ render, resize: () => echarts.getInstanceByDom(containerEl)?.resize() });

  return {
    push(newScore, newLevel) {
      score = newScore;
      level = newLevel;
      render();
    },
    reset() {
      score = 0;
      level = 'low';
      render();
    },
    render,
  };
}

const liveChart = makeRiskGauge(document.getElementById('riskGauge'), document.getElementById('riskPill'));
const uploadChart = makeRiskGauge(document.getElementById('uploadRiskGauge'), document.getElementById('uploadRiskPill'));

// ---- acoustic metric line charts (pitch + 3a's jitter/shimmer/HNR/pause/speech-rate) ----
// Separate from the risk gauge above: these stream live during the call itself (ASR/acoustic/
// emotion, no LLM — pipeline/chunk_worker.py's process_chunk_signals), one point per chunk —
// an actual time series, unlike the risk gauge's single end-of-call reading.
// colorVar: a CSS custom-property name (see index.html's :root) — one hue per metric instead of
// repeating --accent for every tile, so the 3a grid reads as distinct series, not one color
// block six times over. mean_pitch keeps --accent since it's the flagship chart in its own panel.
const METRIC_DEFS = {
  mean_pitch: { label: '音高', unit: 'Hz', min: 60, max: 320, digits: 0, colorVar: '--accent' },
  jitter_local: { label: 'Jitter', unit: '%', min: 0, max: 5, digits: 2, colorVar: '--chart-jitter' },
  shimmer_local: { label: 'Shimmer', unit: '%', min: 0, max: 10, digits: 2, colorVar: '--chart-shimmer' },
  hnr: { label: 'HNR', unit: 'dB', min: 0, max: 30, digits: 1, colorVar: '--chart-hnr' },
  pause_ratio: { label: '停頓佔比', unit: '%', min: 0, max: 90, digits: 1, colorVar: '--chart-pause' },
  speech_rate_variation: { label: '語速變化', unit: '', min: 0, max: 10, digits: 2, colorVar: '--chart-speech-rate' },
};
// 3a's small-multiples grid — pitch keeps its own larger chart in the existing panel, these five
// get the compact grid (see index.html's .metric-grid).
const GRID_METRIC_KEYS = ['jitter_local', 'shimmer_local', 'hnr', 'pause_ratio', 'speech_rate_variation'];

function makeEChartsLineMetric(containerEl, { min, max, unit, digits, colorVar }) {
  let points = [];
  let baselineVal = null;

  function render() {
    const chart = ensureEChart(containerEl);
    const accent = cssVar(colorVar || '--accent');
    chart.setOption(
      {
        grid: { left: 2, right: 2, top: 6, bottom: 2 },
        xAxis: { type: 'category', show: false, boundaryGap: false, data: points.map((_, i) => i) },
        yAxis: { type: 'value', min, max, show: false },
        tooltip: { trigger: 'axis', formatter: (params) => `${Number(params[0].value).toFixed(digits)}${unit}` },
        series: [
          {
            type: 'line',
            data: points,
            showSymbol: false,
            smooth: true,
            lineStyle: { color: accent, width: 2.5 },
            areaStyle: { color: accent, opacity: 0.2 },
            markLine:
              baselineVal == null
                ? undefined
                : {
                    symbol: 'none',
                    silent: true,
                    animation: false,
                    lineStyle: { color: cssVar('--ink-faint'), type: 'dashed', width: 1 },
                    label: { show: false },
                    data: [{ yAxis: baselineVal }],
                  },
          },
        ],
      },
      true
    );
  }

  registerChart({ render, resize: () => echarts.getInstanceByDom(containerEl)?.resize() });

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

function pitchChartContainer(prefix) {
  return document.getElementById(prefix + 'PitchChart');
}

// camelCase DOM-id suffix for a snake_case metric key, e.g. "pause_ratio" -> "PauseRatio".
function metricIdSuffix(key) {
  return key.split('_').map((s) => s.charAt(0).toUpperCase() + s.slice(1)).join('');
}

function metricChartContainer(prefix, key) {
  return document.getElementById(prefix + metricIdSuffix(key) + 'Chart');
}

function metricChartContainerScoped(card, key) {
  return card.querySelector(`.test-clip-${key.replace(/_/g, '-')}-chart`);
}

// Builds one renderer per GRID_METRIC_KEYS entry (+ pitch, handled separately by the caller
// since it lives in a different-sized chart) — shared shape used by createStreamingView.
function makeGridChartRenderers(containerForKey) {
  const renderers = {};
  GRID_METRIC_KEYS.forEach((key) => {
    const def = METRIC_DEFS[key];
    renderers[key] = makeEChartsLineMetric(containerForKey(key), { min: def.min, max: def.max, unit: def.unit, digits: def.digits, colorVar: def.colorVar });
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

// ---- 3c: 8-dim acoustic "deception profile" radar (ECharts native radar chart) —
// visualization only, see index.html's .radar-disclaimer text. Each axis maps 1:1 to a single
// already-streamed acoustic metric (no combining multiple indicators into one axis — that's
// the shape of antifraud_v2's normalization bug, see docs/DESIGN.md §1). Computed entirely
// client-side from data already pushed to this view; never touches pipeline/reasoning code, so
// it cannot leak into risk scoring the way the old system did.
// narrative/metricLabel/unit label each axis and drive the tooltip — order is the order ECharts
// lays the 8 indicators out around the circle.
const RADAR_AXES = [
  { key: 'pitch_instability', invert: false, narrative: '攻擊性語氣', metricLabel: '音高不穩定度', unit: '' },
  { key: 'shimmer_local', invert: false, narrative: '矛盾衝突', metricLabel: 'Shimmer', unit: '%' },
  { key: 'mean_pitch', invert: false, narrative: '明確否認', metricLabel: '平均音高', unit: 'Hz' },
  { key: 'hnr', invert: true, narrative: '尷尬掩蓋', metricLabel: 'HNR', unit: 'dB' }, // lower HNR = more of this indicator
  { key: 'speech_rate_variation', invert: false, narrative: '警覺避談', metricLabel: '語速變化', unit: '' },
  { key: 'pause_ratio', invert: false, narrative: '猶豫不決', metricLabel: '停頓佔比', unit: '%' },
  { key: 'mean_volume', invert: false, narrative: '異常興奮', metricLabel: '平均音量', unit: '' },
  { key: 'jitter_local', invert: false, narrative: '邏輯漏洞', metricLabel: 'Jitter', unit: '%' },
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

const RADAR_NEUTRAL_VALUES8 = [50, 50, 50, 50, 50, 50, 50, 50];

// history: plain object { metricKey: number[] }, accumulated by createStreamingView.onChunk.
// ECharts' own series transition animates between the previous and new values automatically
// (animationDurationUpdate below) — no hand-rolled requestAnimationFrame tweening needed.
function makeEChartsRadar(containerEl) {
  let lastHistory = null;
  let lastBaseline = null;

  function computeValues() {
    if (!lastHistory) return RADAR_NEUTRAL_VALUES8;
    return RADAR_AXES.map((axis) => {
      const vals = lastHistory[axis.key] || [];
      const current = vals.length > 0 ? vals[vals.length - 1] : null;
      const baselineVal = lastBaseline ? lastBaseline[axis.key] : null;
      return zScoreTo100(current, baselineVal, vals, axis.invert);
    });
  }

  function render() {
    const chart = ensureEChart(containerEl);
    const accent = cssVar('--accent');
    const values = computeValues();
    chart.setOption(
      {
        radar: {
          indicator: RADAR_AXES.map((a) => ({ name: a.narrative, min: 0, max: 100 })),
          splitNumber: 4,
          axisName: { color: cssVar('--ink-faint'), fontSize: 10 },
          splitLine: { lineStyle: { color: cssVar('--border') } },
          splitArea: { show: false },
          axisLine: { lineStyle: { color: cssVar('--border') } },
        },
        tooltip: {
          trigger: 'item',
          // Hovering the shape (or any of its vertices) shows every axis's real value at once —
          // denser but simpler than the old per-axis hit-circle hover this replaces.
          formatter: () =>
            RADAR_AXES.map((axis, i) => {
              const vals = (lastHistory && lastHistory[axis.key]) || [];
              const current = vals.length > 0 ? vals[vals.length - 1] : null;
              const raw = current != null ? `${current.toFixed(1)}${axis.unit}` : '尚無資料';
              return `<strong>${axis.narrative}</strong>：${axis.metricLabel} ${raw}（剖面 ${Math.round(values[i])}/100）`;
            }).join('<br>'),
        },
        series: [
          {
            type: 'radar',
            animationDurationUpdate: 250,
            data: [
              {
                value: values,
                areaStyle: { color: accent, opacity: 0.26 },
                lineStyle: { color: accent, width: 2 },
                itemStyle: { color: accent },
              },
            ],
          },
        ],
      },
      true
    );
  }

  registerChart({ render, resize: () => echarts.getInstanceByDom(containerEl)?.resize() });

  return {
    render(history, baseline) {
      lastHistory = history;
      lastBaseline = baseline;
      render();
    },
    reset() {
      lastHistory = null;
      lastBaseline = null;
      render();
    },
  };
}

// ---- eGeMAPS (openSMILE) full-88-dim detail panel — a horizontal bar of each dim's z-score
// vs. this call's own baseline, same self-referential z-score idea as the radar above (50 =
// same as baseline), scaled to all 88 eGeMAPSv02 functionals instead of a curated 8. Mirrors
// audio/egemaps.py's EGEMAPS_HIGHLIGHT_KEYS/EGEMAPS_LABELS_ZH — those 10 get a real label and a
// highlighted bar; the other 78 get a shortened raw functional name (no 88-entry translation
// table) so the full set stays browsable inside the collapsed <details> it lives in
// (index.html's .egemaps-details/.egemaps-scroll — capped height + internal scroll is what
// keeps 88 rows from ever blowing up the page layout).
const EGEMAPS_HIGHLIGHT_KEYS = [
  'loudness_sma3_amean', 'spectralFlux_sma3_amean', 'alphaRatioV_sma3nz_amean',
  'hammarbergIndexV_sma3nz_amean', 'F1frequency_sma3nz_amean', 'F2frequency_sma3nz_amean',
  'mfcc1_sma3_amean', 'slopeV0-500_sma3nz_amean', 'VoicedSegmentsPerSec', 'equivalentSoundLevel_dBp',
];
const EGEMAPS_LABELS_ZH = {
  loudness_sma3_amean: '響度', spectralFlux_sma3_amean: '頻譜變化率', alphaRatioV_sma3nz_amean: 'Alpha 比率',
  hammarbergIndexV_sma3nz_amean: 'Hammarberg 指數', F1frequency_sma3nz_amean: '共振峰 F1',
  F2frequency_sma3nz_amean: '共振峰 F2', mfcc1_sma3_amean: 'MFCC1', 'slopeV0-500_sma3nz_amean': '頻譜斜率（低頻）',
  VoicedSegmentsPerSec: '有聲段/秒', equivalentSoundLevel_dBp: '等效音量',
};

// e.g. "F0semitoneFrom27.5Hz_sma3nz_stddevNorm" -> "F0semitoneFrom27.5Hz · stddevNorm" — strips
// openSMILE's smoothing-filter suffix, keeps the rest scannable without a full translation.
function egemapsFallbackLabel(key) {
  return key.replace(/_sma3nz?/g, '').split('_').join(' · ');
}

function egemapsLabel(key) {
  return EGEMAPS_LABELS_ZH[key] || egemapsFallbackLabel(key);
}

// Groups the 88 dims by what they're physically measuring, reusing the same 6-hue set the 3a
// grid uses (index.html's --accent/--chart-*) — one color per group instead of one flat gray,
// so the detail panel isn't a monochrome wall even before you know which rows matter. Order
// matters (first match wins): mfccV*/F1-3 checked before the plain mfcc*/spectralFlux* fallback.
function egemapsGroupColor(key) {
  if (key.startsWith('F0semitone')) return cssVar('--chart-jitter'); // pitch
  if (key.startsWith('loudness') || key === 'equivalentSoundLevel_dBp' || key === 'loudnessPeaksPerSec') {
    return cssVar('--chart-speech-rate'); // loudness/energy
  }
  if (key.startsWith('F1') || key.startsWith('F2') || key.startsWith('F3')) return cssVar('--chart-pause'); // formants
  if (key.startsWith('jitter') || key.startsWith('shimmer') || key.startsWith('HNR') || key.startsWith('logRelF0')) {
    return cssVar('--chart-shimmer'); // voice quality
  }
  if (key.startsWith('spectralFlux') || key.startsWith('mfcc')) return cssVar('--chart-hnr'); // spectral/timbre
  return cssVar('--accent'); // alphaRatio/hammarberg/slope/voicing-rhythm catch-all
}

const EGEMAPS_ROW_HEIGHT = 15;

function makeEChartsEgemapsBar(containerEl) {
  let lastHistory = null; // { featureKey: number[] }, keyed in server-arrival order (== eGeMAPS's own canonical order)
  let lastBaseline = null; // { featureKey: number } | null
  let keys = [];

  function render() {
    if (keys.length === 0) return; // nothing pushed yet — nothing sensible to draw
    const chart = ensureEChart(containerEl);
    const muted = cssVar('--ink-faint');
    const ink = cssVar('--ink');
    containerEl.style.height = `${keys.length * EGEMAPS_ROW_HEIGHT + 40}px`;
    const isHighlight = (i) => EGEMAPS_HIGHLIGHT_KEYS.includes(keys[i]);
    // Color says "what kind of feature is this" (egemapsGroupColor, one of the same 6 hues the
    // 3a grid uses); opacity + bold label say "is this one of the 10 usually-worth-checking
    // ones" — two independent signals instead of one color meaning both, so the panel isn't a
    // flat gray wall for the 78 non-highlight rows.
    const barColors = keys.map(egemapsGroupColor);
    const values = keys.map((k) => {
      const vals = (lastHistory && lastHistory[k]) || [];
      const current = vals.length > 0 ? vals[vals.length - 1] : null;
      const baselineVal = lastBaseline ? lastBaseline[k] : null;
      return zScoreTo100(current, baselineVal, vals, false);
    });
    chart.setOption(
      {
        grid: { left: 150, right: 30, top: 10, bottom: 10 },
        xAxis: { type: 'value', min: 0, max: 100, show: false },
        yAxis: {
          type: 'category',
          data: keys.map(egemapsLabel),
          inverse: true,
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: {
            fontSize: 11,
            color: (_value, index) => (isHighlight(index) ? ink : muted),
            fontWeight: (_value, index) => (isHighlight(index) ? 700 : 400),
          },
        },
        tooltip: {
          trigger: 'item',
          formatter: (p) => {
            const k = keys[p.dataIndex];
            const vals = (lastHistory && lastHistory[k]) || [];
            const current = vals.length > 0 ? vals[vals.length - 1] : null;
            return `<strong>${egemapsLabel(k)}</strong><br>目前值：${current != null ? current.toFixed(3) : '尚無資料'}<br>剖面分數：${Math.round(p.value)}/100`;
          },
        },
        series: [
          {
            type: 'bar',
            barWidth: 9,
            data: values.map((v, i) => ({
              value: v,
              itemStyle: { color: barColors[i], opacity: isHighlight(i) ? 0.95 : 0.45 },
            })),
          },
        ],
      },
      true
    );
    chart.resize();
  }

  registerChart({ render, resize: () => echarts.getInstanceByDom(containerEl)?.resize() });

  return {
    render(history, baseline) {
      lastHistory = history;
      lastBaseline = baseline;
      if (keys.length === 0 && history) keys = Object.keys(history);
      render();
    },
    reset() {
      lastHistory = null;
      lastBaseline = null;
      keys = [];
      const chart = echarts.getInstanceByDom(containerEl);
      if (chart) chart.clear();
      containerEl.style.height = '';
    },
  };
}

// ---- HTML generator for the 3a/3b/3c panel shared by live/upload/test-clip screens ----
// classIdAttr(prefix, useClass, fixedClasses, name) returns a single class="..." attribute
// (plus id="..." when not useClass) for a bare camelCase element name (e.g. "radarChart",
// "jitterLocalChart") combined with the element's own fixed CSS class(es).
// Returns a single `class="..."` attribute (id-based mode also appends a separate `id="..."`)
// — MUST stay a single class attribute per element: two class="..." attributes on one tag is
// invalid HTML, and browsers silently keep only the first, dropping the second (confirmed by a
// real headless-browser run: this exact bug made every scoped .test-clip-* selector return
// null, which then threw inside the chart renderer's ECharts init on a null container).
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
      <div ${a('metric-chart-wrap', base + 'Chart')}></div>
    </div>`;
  }).join('');

  return `
    <div class="metric-grid">${metricCells}</div>
    <div ${a('baseline-readout mono', 'baselineReadout')}>尚未建立基準值（通話開頭 15 秒後自動建立）。</div>
    <div class="radar-wrap">
      <div ${a('radar-chart', 'radarChart')}></div>
      <div class="radar-disclaimer">聲學特徵剖面（僅供參考，非風險判定）— 與本通話自己的開頭基準值比較</div>
    </div>
    <details class="egemaps-details">
      <summary class="egemaps-summary">進階聲學特徵（eGeMAPS，共 88 維）</summary>
      <div class="egemaps-note">openSMILE eGeMAPSv02 標準特徵集，數值為與本通話開頭基準值的差異程度（50 = 與基準相同）；深色列是較常參考的 10 項，其餘為完整 88 維供查閱。</div>
      <div class="egemaps-scroll"><div ${a('egemaps-chart', 'egemapsChart')}></div></div>
    </details>`;
}

// ---- emotion badge + acoustic readout ----
// Covers both backends' label vocabularies (audio/emotion.py dispatches between them per the
// "emotion_backend" setting) — TIMNet's anger/boredom/disgust/fear/happy/neutral/sad and
// WavLM's anger/contempt/disgust/fear/happy/neutral/sad/surprise/other (already lowercased/
// normalized server-side, see detectors/emotion_wavlm.py's _LABEL_MAP) share every overlapping
// key, so only "boredom" (TIMNet-only) and "contempt"/"surprise"/"other" (WavLM-only) differ.
const EMOTION_LABELS_ZH = {
  anger: '生氣', boredom: '無聊', disgust: '厭惡', fear: '恐懼', happy: '開心', neutral: '中性', sad: '難過',
  contempt: '輕蔑', surprise: '驚訝', other: '其他',
};
const FRAUD_TYPE_LABELS_ZH = {
  investment_fraud: '投資詐騙', phishing_fraud: '網路釣魚詐騙', identity_theft: '身分冒用',
  lottery_fraud: '中獎摸彩詐騙', banking_fraud: '銀行詐騙', extortion_fraud: '勒索詐騙',
  customer_service_fraud: '客服詐騙', unclassified: '未分類',
};

function emotionBadgeClass(label) {
  if (label === 'happy') return 'emo-low';
  if (label === 'fear' || label === 'sad' || label === 'surprise') return 'emo-medium';
  if (label === 'anger' || label === 'disgust' || label === 'contempt') return 'emo-high';
  return 'emo-neutral'; // neutral, boredom, other
}

function formatAcousticReadout(a) {
  const pitch = a.mean_pitch != null ? `${a.mean_pitch.toFixed(0)} Hz` : '無足夠有聲段';
  return `音高 ${pitch} · jitter ${a.jitter_local}% · shimmer ${a.shimmer_local}% · HNR ${a.hnr}dB · 停頓 ${a.pause_ratio}%`;
}

// msg.fake_score/ai_voice_flag/fusion_source come straight off SynthesizeResult (server/ws.py,
// server/upload.py's final_analysis event) — reasoning/fusion.py's fuse() already computed
// which of Line 1 (acoustic: AI cloned voice) or Line 2 (semantic: scam-script content) drove
// the verdict, but until now the frontend only ever read msg.fraud_type here, silently dropping
// the other three fields it was already being sent. Showing them as two separate badges is what
// distinguishes "sounds like a cloned voice" from "content reads like a scam script" instead of
// one undifferentiated risk chip.
// The other chips below interpolate only server-controlled enum labels, but the patent chips
// carry LLM-produced transcript quotes — those go through here rather than into raw HTML.
function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (ch) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
  ));
}

function badgeRowHTML(msg) {
  const chips = [];
  // Always shown as a plain score, not gated on ai_voice_flag/threshold — a continuous "how
  // AI-like did this sound" readout, not a pop-up warning. Uses .info-badge.line1's teal accent
  // styling only past the actual fusion threshold (msg.ai_voice_flag true, i.e. the same
  // fake_score >= DEEPFAKE_FAKE_SCORE_THRESHOLD that made Line 1 win fusion) — the real "this is
  // dangerous" signal still lives in the alert banner and 最終研判 text, not this chip's color.
  if (msg.fake_score != null) {
    chips.push(
      `<span class="info-badge${msg.ai_voice_flag ? ' line1' : ''}">聲學：AI 合成／複製語音機率 ${Math.round(msg.fake_score * 100)}%</span>`
    );
  }
  if (msg.fusion_source === 'line2_semantic' && msg.fraud_type && msg.fraud_type.fraud_type !== 'unclassified') {
    chips.push(`<span class="info-badge line2">文字語意：疑似詐騙話術</span>`);
  }
  if (msg.fraud_type && msg.fraud_type.fraud_type && msg.fraud_type.fraud_type !== 'unclassified') {
    const label = FRAUD_TYPE_LABELS_ZH[msg.fraud_type.fraud_type] || msg.fraud_type.fraud_type;
    chips.push(`<span class="info-badge">疑似：${label}（${msg.fraud_type.confidence}）</span>`);
  }
  if (msg.audio_quality && msg.audio_quality.narrowband) {
    chips.push(`<span class="info-badge narrowband">電話頻寬（窄頻）— 聲學細節可信度較低</span>`);
  }
  // 專利 TW I904863 S312 的詐騙模式標記。只有在 fuse() 判定成立時後端才會產生內容（見
  // pipeline/chunk_worker.py 的 gating），所以這裡不需要再判斷風險等級——有值就代表已判定成立。
  // 可複選，每個模式一個 chip；title 帶出命中的逐字稿引句，滑過去就能看到判斷依據。
  (msg.matched_patterns || []).forEach((p) => {
    const quotes = (p.quotes || []).join('｜');
    const tip = quotes ? `依據：${quotes}` : '無逐字稿引句';
    chips.push(`<span class="info-badge patent" title="${escapeHTML(tip)}">${escapeHTML(p.name)}</span>`);
  });
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
// on the live screen (which uses the top alert banner and the settled risk gauge/pill instead).
function createStreamingView(els) {
  let alerts = [];
  let baseline = null;
  let egemapsBaseline = null;
  const metricHistory = { mean_pitch: [], pitch_instability: [], mean_volume: [] }; // radar-only trackers
  const egemapsHistory = {}; // { featureKey: number[] } — lazily keyed from the first msg.egemaps

  return {
    reset() {
      alerts = [];
      baseline = null;
      egemapsBaseline = null;
      Object.keys(metricHistory).forEach((k) => { metricHistory[k] = []; });
      Object.keys(egemapsHistory).forEach((k) => delete egemapsHistory[k]);
      els.transcriptList.innerHTML = '';
      els.pitchChart.reset();
      if (els.gridCharts) Object.values(els.gridCharts).forEach((c) => c.reset());
      if (els.baselineReadout) els.baselineReadout.textContent = formatBaselineReadout(null);
      if (els.radar) els.radar.reset();
      if (els.egemapsChart) els.egemapsChart.reset();
      if (els.badgeRow) els.badgeRow.innerHTML = '';
      els.acousticReadout.textContent = '尚未偵測到聲音。';
      els.emotionBadge.textContent = '尚無資料';
      els.emotionBadge.className = 'emo-badge emo-neutral';
      if (els.deepfakeBadge) {
        els.deepfakeBadge.textContent = 'AI 合成 --%';
        els.deepfakeBadge.className = 'deepfake-badge';
      }
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
      if (msg.egemaps && els.egemapsChart) {
        Object.entries(msg.egemaps).forEach(([k, v]) => {
          if (!egemapsHistory[k]) egemapsHistory[k] = [];
          egemapsHistory[k].push(v);
        });
        els.egemapsChart.render(egemapsHistory, egemapsBaseline);
      }
      if (msg.baseline) {
        baseline = msg.baseline;
        els.pitchChart.setBaseline(baseline.mean_pitch);
        if (els.gridCharts) GRID_METRIC_KEYS.forEach((key) => els.gridCharts[key].setBaseline(baseline[key]));
        if (els.baselineReadout) els.baselineReadout.textContent = formatBaselineReadout(baseline);
      }
      if (msg.egemaps_baseline) egemapsBaseline = msg.egemaps_baseline;
      if (msg.emotion) {
        els.emotionBadge.textContent = `${EMOTION_LABELS_ZH[msg.emotion.label] || msg.emotion.label} ${msg.emotion.top_prob}`;
        els.emotionBadge.className = 'emo-badge ' + emotionBadgeClass(msg.emotion.label);
      }
      if (msg.deepfake && els.deepfakeBadge) {
        els.deepfakeBadge.textContent = `AI 合成 ${Math.round(msg.deepfake.fake_score * 100)}%`;
        els.deepfakeBadge.className = 'deepfake-badge' + (msg.deepfake.fake_score >= 0.85 ? ' deepfake-notable' : '');
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

// pipeline/call_state.py's apply_final_result() already distinguishes three alert reasons
// (deepfake_voice / hard_trigger / final_analysis) — this used to collapse the latter two into
// the same generic "風險持續偏高" title, which misleadingly implied every alert was a vague
// "risk trending up" read when it could just as easily be Line 1 (AI cloned voice, an acoustic
// signal) or Line 2 (scam-script content, a semantic signal). Surfacing the real reason here.
function alertTitle(a) {
  if (a.reason === 'hard_trigger') return `高信度示警 · ${a.trigger_name || ''}`;
  if (a.reason === 'deepfake_voice') return '示警 · 偵測到 AI 合成／複製語音（聲學訊號）';
  if (a.reason === 'final_analysis') return '示警 · 話術內容疑似詐騙（文字語意訊號）';
  return '示警 · 風險持續偏高';
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
        <div class="alert-title">${alertTitle(a)}</div>
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
  document.querySelector('#alertBanner .alert-title').textContent = alertTitle(alertMsg);
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
  pitchChart: makeEChartsLineMetric(pitchChartContainer('live'), { min: METRIC_DEFS.mean_pitch.min, max: METRIC_DEFS.mean_pitch.max, unit: METRIC_DEFS.mean_pitch.unit, digits: METRIC_DEFS.mean_pitch.digits }),
  gridCharts: makeGridChartRenderers((key) => metricChartContainer('live', key)),
  acousticReadout: document.getElementById('liveAcousticReadout'),
  emotionBadge: document.getElementById('liveEmotionBadge'),
  deepfakeBadge: document.getElementById('liveDeepfakeBadge'),
  baselineReadout: document.getElementById('liveBaselineReadout'),
  radar: makeEChartsRadar(document.getElementById('liveRadarChart')),
  egemapsChart: makeEChartsEgemapsBar(document.getElementById('liveEgemapsChart')),
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
  const deleteAllBtn = document.getElementById('btnHistoryDeleteAll');
  let calls;
  try {
    calls = await (await fetch('/api/calls')).json();
  } catch (err) {
    container.innerHTML = `<div class="placeholder-note">無法載入歷史紀錄：${err.message}</div>`;
    deleteAllBtn.style.display = 'none';
    return;
  }
  deleteAllBtn.style.display = calls.length ? 'inline-block' : 'none';
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
      <button class="history-delete" data-call-id="${c.id}" title="刪除這筆紀錄" aria-label="刪除">&times;</button>
    </div>`
    )
    .join('');
}

async function deleteCall(callId) {
  try {
    const res = await fetch(`/api/calls/${callId}`, { method: 'DELETE' });
    if (!res.ok && res.status !== 404) throw new Error(`伺服器回應 ${res.status}`);
  } catch (err) {
    alert('刪除失敗：' + err.message);
    return;
  }
  loadHistory();
}

document.getElementById('btnHistoryDeleteAll').addEventListener('click', async () => {
  if (!confirm('確定要刪除全部通話紀錄嗎？此操作無法復原。')) return;
  try {
    const res = await fetch('/api/calls', { method: 'DELETE' });
    if (!res.ok) throw new Error(`伺服器回應 ${res.status}`);
  } catch (err) {
    alert('刪除失敗：' + err.message);
    return;
  }
  loadHistory();
});

// Delegated once on the container (not inside loadHistory, which replaces its innerHTML on
// every tab visit / refresh — attaching there would stack up a duplicate listener per visit).
document.getElementById('historyList').addEventListener('click', (e) => {
  const delBtn = e.target.closest('.history-delete');
  if (delBtn) {
    e.stopPropagation();
    if (confirm('確定要刪除這筆通話紀錄嗎？此操作無法復原。')) deleteCall(delBtn.dataset.callId);
    return;
  }
  const row = e.target.closest('.history-row');
  if (row) openHistoryDetail(row.dataset.callId);
});

const historyChart = makeRiskGauge(document.getElementById('historyRiskGauge'), document.getElementById('historyRiskPill'));

let currentHistoryCallId = null;

async function openHistoryDetail(callId) {
  currentHistoryCallId = callId;
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

document.getElementById('btnHistoryDelete').addEventListener('click', async () => {
  if (!currentHistoryCallId) return;
  if (!confirm('確定要刪除這筆通話紀錄嗎？此操作無法復原。')) return;
  await deleteCall(currentHistoryCallId);
  document.getElementById('historyDetailView').style.display = 'none';
  document.getElementById('historyListView').style.display = 'block';
});

// ---- settings (server/api.py: GET/POST /api/settings) ----
const providerSelect = document.getElementById('providerSelect');
const modelSelect = document.getElementById('modelSelect');
const line2Select = document.getElementById('line2Select');
const asrSelect = document.getElementById('asrSelect');
const emotionBackendSelect = document.getElementById('emotionBackendSelect');
const effortRow = document.getElementById('effortRow');
const llmSummaryRow = document.getElementById('llmSummaryRow');
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
  line2Select.value = settings.line2_backend;
  asrSelect.value = settings.asr_backend;
  emotionBackendSelect.value = settings.emotion_backend;
  effortRow.querySelectorAll('.effort-opt').forEach((opt) =>
    opt.classList.toggle('active', opt.dataset.value === settings.llm_effort)
  );
  llmSummaryRow.querySelectorAll('.effort-opt').forEach((opt) =>
    opt.classList.toggle('active', opt.dataset.value === (settings.llm_final_summary_enabled ? 'on' : 'off'))
  );
  currentHardTriggers = [...settings.hard_triggers];
  renderHardTriggers();

  // Also reflect the active model/effort in the live-call header chip.
  document.querySelector('.model-chip').textContent = `${settings.llm_model} · ${settings.llm_effort}`;
}

providerSelect.addEventListener('change', () => saveSettings({ llm_provider: providerSelect.value }));
modelSelect.addEventListener('change', () => saveSettings({ llm_model: modelSelect.value }));
line2Select.addEventListener('change', () => saveSettings({ line2_backend: line2Select.value }));
asrSelect.addEventListener('change', () => saveSettings({ asr_backend: asrSelect.value }));
emotionBackendSelect.addEventListener('change', () => saveSettings({ emotion_backend: emotionBackendSelect.value }));
effortRow.querySelectorAll('.effort-opt').forEach((opt) =>
  opt.addEventListener('click', () => {
    effortRow.querySelectorAll('.effort-opt').forEach((o) => o.classList.remove('active'));
    opt.classList.add('active');
    saveSettings({ llm_effort: opt.dataset.value });
  })
);
llmSummaryRow.querySelectorAll('.effort-opt').forEach((opt) =>
  opt.addEventListener('click', () => {
    llmSummaryRow.querySelectorAll('.effort-opt').forEach((o) => o.classList.remove('active'));
    opt.classList.add('active');
    saveSettings({ llm_final_summary_enabled: opt.dataset.value === 'on' });
  })
);

// ---- upload analysis (server/upload.py: POST /api/upload-call, streamed NDJSON) ----
const uploadView = createStreamingView({
  transcriptList: document.getElementById('uploadTranscriptList'),
  pitchChart: makeEChartsLineMetric(pitchChartContainer('upload'), { min: METRIC_DEFS.mean_pitch.min, max: METRIC_DEFS.mean_pitch.max, unit: METRIC_DEFS.mean_pitch.unit, digits: METRIC_DEFS.mean_pitch.digits }),
  gridCharts: makeGridChartRenderers((key) => metricChartContainer('upload', key)),
  acousticReadout: document.getElementById('uploadAcousticReadout'),
  emotionBadge: document.getElementById('uploadEmotionBadge'),
  deepfakeBadge: document.getElementById('uploadDeepfakeBadge'),
  baselineReadout: document.getElementById('uploadBaselineReadout'),
  radar: makeEChartsRadar(document.getElementById('uploadRadarChart')),
  egemapsChart: makeEChartsEgemapsBar(document.getElementById('uploadEgemapsChart')),
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
  // Reveal the container BEFORE reset() — reset() is what lazily calls echarts.init() on each
  // chart the first time, and a container that's still display:none measures as zero width, so
  // ECharts would size its canvas to 0px and never fix itself without an explicit later resize().
  document.getElementById('uploadResult').style.display = 'block'; // shown immediately — chunk_update events fill it in progressively
  uploadView.reset();
  uploadChart.reset();

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
// Was 3 — silently hid any clip that didn't sort alphabetically into the first 3 per category
// (server/testdata.py's list_test_clips() returns them sorted by filename). The corpus is now
// 13 scam + 10 benign (docs/DESIGN.md §6's own 10-20-per-category target), so cap high enough
// that nothing gets hidden just for having a filename late in the alphabet.
const TEST_CLIP_COUNT = 20;

function testClipCardHTML(category, filename) {
  return `
  <div class="panel test-clip-card" data-category="${category}" data-filename="${filename}">
    <div class="panel-head">
      <div class="panel-title">${filename}</div>
      <button class="btn-toggle-listen test-clip-run">執行分析</button>
    </div>
    <audio controls preload="none" src="/api/test-clips/${category}/${filename}/audio"></audio>
    <div class="test-clip-result" style="display:none;">
      <div class="field-help mono test-clip-summary"></div>
      <div class="badge-row test-clip-badge-row"></div>
      <div class="panel-head" style="margin-top:10px; margin-bottom:6px;">
        <div class="panel-title" style="font-size:11px;">聲學／情緒</div>
        <div style="display:flex; gap:6px; align-items:center;">
          <span class="deepfake-badge test-clip-deepfake-badge" title="AI 合成／複製語音機率——連續分數，非警告">AI 合成 --%</span>
          <span class="emo-badge emo-neutral test-clip-emotion-badge">尚無資料</span>
        </div>
      </div>
      <div class="pitch-chart-wrap test-clip-pitch-chart"></div>
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
// cards can be on screen at once, see metricChartContainerScoped's docstring for why the chart
// containers are looked up via card.querySelector() instead of a global id).
function createTestClipView(card) {
  return createStreamingView({
    transcriptList: card.querySelector('.test-clip-transcript'),
    pitchChart: makeEChartsLineMetric(card.querySelector('.test-clip-pitch-chart'), {
      min: METRIC_DEFS.mean_pitch.min, max: METRIC_DEFS.mean_pitch.max, unit: METRIC_DEFS.mean_pitch.unit, digits: METRIC_DEFS.mean_pitch.digits,
    }),
    gridCharts: makeGridChartRenderers((key) => metricChartContainerScoped(card, key)),
    acousticReadout: card.querySelector('.test-clip-acoustic-readout'),
    emotionBadge: card.querySelector('.test-clip-emotion-badge'),
    deepfakeBadge: card.querySelector('.test-clip-deepfake-badge'),
    baselineReadout: card.querySelector('.test-clip-baseline-readout'),
    radar: makeEChartsRadar(card.querySelector('.test-clip-radar-chart')),
    egemapsChart: makeEChartsEgemapsBar(card.querySelector('.test-clip-egemaps-chart')),
    badgeRow: card.querySelector('.test-clip-badge-row'),
    alertsList: card.querySelector('.test-clip-alerts'),
    finalSummary: card.querySelector('.test-clip-final-summary'),
    evidence: card.querySelector('.test-clip-evidence'),
  });
}

// server/testdata.py discovers categories from eval/test_clips/'s subdirectories rather than
// a hardcoded list — this map is just display labels for the ones that exist today; an unknown
// future category (e.g. real human-voice clips, or clips grouped by emotion) still renders
// fine, just with its raw folder name as the label.
const CATEGORY_LABELS_ZH = { benign: 'Benign（正常通話）', scam: 'Scam（詐騙通話）' };

function categoryLabel(category) {
  return CATEGORY_LABELS_ZH[category] || category;
}

// Each category renders as a collapsed <details> section (only the first starts open) — with
// more than two categories the old flat "one long list" layout would make the page unusably
// long, so the page grows downward as closed summaries instead.
async function loadTestData() {
  const container = document.getElementById('testDataCategories');
  let clips;
  try {
    clips = await (await fetch('/api/test-clips')).json();
  } catch (err) {
    container.innerHTML = `<div class="placeholder-note">無法載入測試音檔清單：${err.message}</div>`;
    return;
  }
  const categories = Object.keys(clips);
  if (categories.length === 0) {
    container.innerHTML = '<div class="placeholder-note">尚未找到任何測試音檔（eval/test_clips/ 底下沒有分類資料夾）。</div>';
    return;
  }
  container.innerHTML = categories
    .map((category, i) => {
      const files = clips[category].slice(0, TEST_CLIP_COUNT);
      return `
      <details class="test-category"${i === 0 ? ' open' : ''}>
        <summary class="test-category-summary"><span>${categoryLabel(category)}</span><span class="test-category-count">${files.length}</span></summary>
        <div class="test-category-body">${files.map((f) => testClipCardHTML(category, f)).join('')}</div>
      </details>`;
    })
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

  // Reveal the container BEFORE reset() — see analyzeUpload's matching comment: reset() is what
  // lazily calls echarts.init() on each chart the first time, and a still-display:none container
  // measures as zero width, so the chart's canvas would be sized to 0px and never self-correct.
  resultEl.style.display = 'block'; // shown immediately — chunk_update events fill it in progressively
  const view = createTestClipView(card);
  view.reset();

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
