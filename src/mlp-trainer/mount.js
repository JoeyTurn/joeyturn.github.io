// src/mlp-trainer/mount.js
import { trainMLP, GRAB_TARGETS } from "./trainer.js";

/* ============================================================
   tiny DOM helpers
   ============================================================ */
function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "style") n.setAttribute("style", v);
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, String(v));
  }
  for (const kid of kids) {
    if (kid == null) continue;
    n.append(typeof kid === "string" ? document.createTextNode(kid) : kid);
  }
  return n;
}

/* ============================================================
   Log/linear slider helpers
   ============================================================ */
function logSliderValue(rawVal, min, max) {
  return min * Math.pow(max / min, rawVal);
}
function valueToLogSlider(val, min, max) {
  return Math.log(val / min) / Math.log(max / min);
}
function fmt(v, sig = 3) {
  if (!isFinite(v)) return String(v);
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 100)  return v.toFixed(0);
  if (Math.abs(v) >= 1)    return v.toPrecision(sig);
  return v.toExponential(2);
}

/* ============================================================
   CSS injected once
   ============================================================ */
function injectStyles() {
  if (document.getElementById("mlp-trainer-styles")) return;
  const s = document.createElement("style");
  s.id = "mlp-trainer-styles";
  s.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    .mlpt-root {
      font-family: 'DM Sans', sans-serif;
      background: transparent;
      color: inherit;
      padding: 4px 0;
      box-sizing: border-box;
    }

    /* ---- Cards ---- */
    .mlpt-card {
      background: color-mix(in srgb, currentColor 4%, transparent);
      border: 1px solid color-mix(in srgb, currentColor 12%, transparent);
      border-radius: 12px;
      padding: 14px 16px;
      margin-bottom: 12px;
    }
    .mlpt-card-title {
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      opacity: 0.38;
      margin-bottom: 12px;
    }

    /* ---- Slider rows ---- */
    .mlpt-slider-grid {
      display: grid;
      grid-template-columns: 120px 1fr 68px;
      align-items: center;
      gap: 7px 10px;
    }
    .mlpt-slider-label {
      font-size: 0.78rem;
      opacity: 0.65;
      font-family: 'IBM Plex Mono', monospace;
      white-space: nowrap;
    }
    .mlpt-slider-val {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.78rem;
      text-align: right;
      opacity: 0.9;
    }

    /* Range slider — thumb color adapts via accent classes */
    input[type=range].mlpt-range {
      -webkit-appearance: none;
      width: 100%;
      height: 3px;
      background: color-mix(in srgb, currentColor 16%, transparent);
      border-radius: 4px;
      outline: none;
      cursor: pointer;
    }
    input[type=range].mlpt-range::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 13px; height: 13px;
      border-radius: 50%;
      background: #7c6af5;
      cursor: pointer;
      box-shadow: 0 0 0 3px color-mix(in srgb, #7c6af5 22%, transparent);
      transition: transform 0.12s;
    }
    input[type=range].mlpt-range:hover::-webkit-slider-thumb { transform: scale(1.22); }
    input[type=range].mlpt-range::-moz-range-thumb {
      width: 13px; height: 13px; border-radius: 50%;
      background: #7c6af5; cursor: pointer; border: none;
    }
    input[type=range].mlpt-range.acc2::-webkit-slider-thumb { background: #4ec9a8; }
    input[type=range].mlpt-range.acc2::-moz-range-thumb   { background: #4ec9a8; }
    input[type=range].mlpt-range.acc3::-webkit-slider-thumb { background: #f0a060; }
    input[type=range].mlpt-range.acc3::-moz-range-thumb   { background: #f0a060; }

    /* ---- Eigval grid ---- */
    .mlpt-eigval-grid { display: grid; gap: 5px 12px; }

    /* ---- Target formula display ---- */
    .mlpt-target-box {
      background: color-mix(in srgb, currentColor 5%, transparent);
      border: 1px solid color-mix(in srgb, currentColor 10%, transparent);
      border-radius: 9px;
      padding: 10px 13px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.78rem;
      line-height: 1.75;
      margin-top: 8px;
    }
    .mlpt-target-box .hlA  { color: #7c6af5; }
    .mlpt-target-box .hlB  { color: #4ec9a8; }
    .mlpt-target-box .dim  { opacity: 0.42;  }

    /* ---- Select ---- */
    select.mlpt-select {
      background: color-mix(in srgb, currentColor 7%, transparent);
      color: inherit;
      border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
      border-radius: 7px;
      padding: 4px 9px;
      font-size: 0.78rem;
      font-family: 'DM Sans', sans-serif;
      cursor: pointer;
      outline: none;
    }
    select.mlpt-select:focus { border-color: #7c6af5; }

    /* ---- Buttons ---- */
    .mlpt-btn {
      padding: 7px 18px;
      border-radius: 9px;
      border: none;
      font-family: 'DM Sans', sans-serif;
      font-size: 0.83rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.13s;
    }
    .mlpt-btn-primary { background: #7c6af5; color: #fff; }
    .mlpt-btn-primary:hover { background: #9080ff; transform: translateY(-1px); }
    .mlpt-btn-secondary {
      background: color-mix(in srgb, currentColor 8%, transparent);
      color: inherit;
      border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
    }
    .mlpt-btn-secondary:hover { border-color: #7c6af5; }
    .mlpt-btn-danger {
      background: color-mix(in srgb, currentColor 8%, transparent);
      color: #e06060;
      border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
    }
    .mlpt-btn-danger:hover { border-color: #e06060; }

    /* ---- Canvas ---- */
    .mlpt-canvas {
      width: 100%;
      border-radius: 9px;
      display: block;
    }

    /* ---- Grab panel ---- */
    .mlpt-grab-row { display: flex; flex-wrap: wrap; gap: 7px; align-items: center; }
    .mlpt-grab-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
    .mlpt-grab-table td {
      border-bottom: 1px solid color-mix(in srgb, currentColor 8%, transparent);
      padding: 4px 7px; opacity: 0.7;
    }
    .mlpt-grab-table td:last-child { text-align: right; opacity: 1; }

    /* ---- Latest grabs ---- */
    .mlpt-latest-table {
      width: 100%; border-collapse: collapse;
      font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace;
    }
    .mlpt-latest-table td {
      border-bottom: 1px solid color-mix(in srgb, currentColor 8%, transparent);
      padding: 3px 7px;
    }
    .mlpt-latest-table td:first-child { opacity: 0.42; }

    /* ---- Status / error lines ---- */
    .mlpt-errline {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.73rem; color: #e06060;
      min-height: 14px;
    }
    .mlpt-stepctr {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.76rem; opacity: 0.48;
      min-height: 14px; margin-bottom: 2px;
    }

    /* ---- Two column grid ---- */
    .mlpt-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 640px) { .mlpt-2col { grid-template-columns: 1fr; } }

    /* ---- Separator ---- */
    .mlpt-sep {
      height: 1px;
      background: color-mix(in srgb, currentColor 10%, transparent);
      margin: 10px 0;
    }

    /* ---- small number input ---- */
    input.mlpt-numbox {
      width: 60px;
      background: color-mix(in srgb, currentColor 7%, transparent);
      color: inherit;
      border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
      border-radius: 6px;
      padding: 3px 7px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.76rem;
      text-align: center;
    }
    input.mlpt-numbox:focus { outline: none; border-color: #7c6af5; }
  `;
  document.head.appendChild(s);
}

/* ============================================================
   Plot rendering — dense ticks, larger canvas
   ============================================================ */
function applyEma(rawArr, alpha) {
  if (!rawArr || rawArr.length === 0) return [];
  const out = new Array(rawArr.length);
  out[0] = rawArr[0];
  for (let i = 1; i < rawArr.length; i++) {
    out[i] = alpha * out[i - 1] + (1 - alpha) * rawArr[i];
  }
  return out;
}

function niceTicksBetween(lo, hi, targetCount) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / targetCount;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  const step = nice * mag;
  const first = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let t = first; t <= hi + 1e-9 * step; t += step) {
    ticks.push(parseFloat(t.toPrecision(10)));
    if (ticks.length > targetCount + 4) break;
  }
  return ticks;
}

function drawPlot(canvas, ys, steps, title, colorHex = "#7c6af5") {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width / dpr;
  const H = canvas.height / dpr;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(dpr, dpr);

  const cs = getComputedStyle(canvas);
  const textColor = cs.color || "#888";
  const gridAlpha  = 0.10;
  const axisAlpha  = 0.25;

  const padL = 66, padB = 46, padT = 32, padR = 16;
  const left = padL, right = W - padR, top = padT, bottom = H - padB;

  // title
  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.fillStyle = textColor;
  ctx.globalAlpha = 0.48;
  ctx.textAlign = "left";
  ctx.fillText(title, left + 2, top - 10);
  ctx.globalAlpha = 1;

  // axes
  ctx.strokeStyle = textColor;
  ctx.globalAlpha = axisAlpha;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top); ctx.lineTo(left, bottom); ctx.lineTo(right, bottom);
  ctx.stroke();
  ctx.globalAlpha = 1;

  if (!ys || ys.length < 2) { ctx.restore(); return; }

  const maxPoints = 1200;
  const stride = Math.max(1, Math.floor(ys.length / maxPoints));
  const pts = [], xpts = [];
  for (let i = 0; i < ys.length; i += stride) {
    if (isFinite(ys[i])) { pts.push(ys[i]); xpts.push(steps ? steps[i] : i); }
  }
  if (pts.length < 2) { ctx.restore(); return; }

  let ymin = Infinity, ymax = -Infinity;
  for (const v of pts) { if (v < ymin) ymin = v; if (v > ymax) ymax = v; }
  if (!(ymax > ymin)) ymax = ymin + 1e-6;
  const ypad = (ymax - ymin) * 0.07;
  const ylo = ymin - ypad, yhi = ymax + ypad;

  const yToPix = (y) => bottom - ((y - ylo) / (yhi - ylo)) * (bottom - top);

  // ---- Y ticks (~7) ----
  const yTicks = niceTicksBetween(ylo, yhi, 7);
  ctx.font = "10px 'IBM Plex Mono', monospace";
  ctx.textAlign = "right";
  for (const t of yTicks) {
    const py = yToPix(t);
    if (py < top - 4 || py > bottom + 4) continue;
    // grid
    ctx.strokeStyle = textColor;
    ctx.globalAlpha = gridAlpha;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    // label
    ctx.fillStyle = textColor;
    ctx.globalAlpha = 0.48;
    const lbl = Math.abs(t) >= 0.01 ? t.toPrecision(3) : t.toExponential(1);
    ctx.fillText(lbl, left - 5, py + 3.5);
    ctx.globalAlpha = 1;
  }

  // ---- X axis (log scale) ----
  const logx = (v) => Math.log1p(Math.max(0, v));
  let xMin = Infinity, xMax = -Infinity;
  for (const xv of xpts) { if (xv < xMin) xMin = xv; if (xv > xMax) xMax = xv; }
  const lxMin = logx(xMin), lxMax = logx(xMax);
  const lxDen = lxMax > lxMin ? lxMax - lxMin : 1e-9;
  const xToPix = (x) => left + ((logx(x) - lxMin) / lxDen) * (right - left);

  // x tick candidates: powers of 10 + 2× + 5× within range
  const xTickSet = new Set();
  for (let p = 0; p <= 6; p++) {
    for (const mult of [1, 2, 3, 5]) {
      const v = mult * Math.pow(10, p);
      if (v >= xMin * 0.9 && v <= xMax * 1.1) xTickSet.add(Math.round(v));
    }
  }
  xTickSet.add(Math.round(xMin));
  xTickSet.add(Math.round(xMax));
  const xTicks = [...xTickSet].sort((a, b) => a - b);

  ctx.font = "10px 'IBM Plex Mono', monospace";
  for (let i = 0; i < xTicks.length; i++) {
    const x = xTicks[i];
    const px = xToPix(x);
    if (px < left - 2 || px > right + 2) continue;
    // vertical grid line
    ctx.strokeStyle = textColor;
    ctx.globalAlpha = gridAlpha;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    // label
    ctx.fillStyle = textColor;
    ctx.globalAlpha = 0.45;
    ctx.textAlign = i === 0 ? "left" : (i === xTicks.length - 1 ? "right" : "center");
    ctx.fillText(x >= 1000 ? (x / 1000) + "k" : String(x), px, bottom + 14);
    ctx.globalAlpha = 1;
  }
  // axis labels
  ctx.globalAlpha = 0.30;
  ctx.textAlign = "center";
  ctx.fillText("step (log scale)", (left + right) / 2, bottom + 28);
  ctx.save();
  ctx.translate(12, (top + bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("MSE loss", 0, 0);
  ctx.restore();
  ctx.globalAlpha = 1;

  // ---- Curve ----
  ctx.shadowColor = colorHex + "55";
  ctx.shadowBlur = 9;
  ctx.strokeStyle = colorHex;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const x = xToPix(xpts[i]);
    const y = yToPix(pts[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  ctx.restore();
}

/* ============================================================
   Grab panel helpers
   ============================================================ */
function fmtVal(v) {
  if (Array.isArray(v)) return `[${v.map((x) => Number(x).toFixed(4)).join(", ")}]`;
  if (typeof v === "number") return Number(v).toFixed(5);
  return String(v);
}

function renderGrabLatest(container, latest) {
  container.innerHTML = "";
  if (!latest) return;
  const table = el("table", { class: "mlpt-latest-table" });
  for (const [k, v] of Object.entries(latest)) {
    table.append(el("tr", {}, el("td", {}, k), el("td", {}, fmtVal(v))));
  }
  container.append(table);
}

const SUMMARY_BY_TARGET = {
  weight: ["rms", "fro_norm", "spectral_norm", "trace_gram", "topk_singular_values", "topk_psd_eigenvalues"],
  activation: ["rms", "fro_norm", "spectral_norm", "trace_gram", "topk_singular_values", "topk_psd_eigenvalues"],
  gradient: ["rms", "grad_global_l2", "grad_per_layer_rms"],
};

function repopulateSummaries(targetSel, summarySel) {
  const t = targetSel.value;
  const opts = SUMMARY_BY_TARGET[t] ?? [];
  const prev = summarySel.value;
  summarySel.innerHTML = "";
  for (const s of opts) summarySel.appendChild(el("option", { value: s }, s));
  if (opts.includes(prev)) summarySel.value = prev;
}

function scalarize(v) {
  if (typeof v === "number") return v;
  if (Array.isArray(v) && v.length) return Number(v[0]);
  return NaN;
}

/* ============================================================
   Slider factory
   KEY FIX: step="any" prevents browser from snapping float
   positions to step grid, which caused "1.02" / "1.98" display.
   Integer sliders round in their onChange handler.
   ============================================================ */
function makeSlider({ label, min, max, value, isLog = false, accent = "", onChange }) {
  const initRaw = isLog
    ? valueToLogSlider(value, min, max)
    : (value - min) / (max - min);

  const sliderEl = el("input", {
    type: "range",
    class: "mlpt-range" + (accent ? " " + accent : ""),
    min: "0", max: "1",
    step: "any",   // ← critical: no browser step-snapping
    value: String(initRaw),
  });

  const valDisplay = el("div", { class: "mlpt-slider-val" });

  function getCurrent() {
    const raw = parseFloat(sliderEl.value);
    return isLog ? logSliderValue(raw, min, max) : min + raw * (max - min);
  }

  sliderEl.addEventListener("input", () => {
    const v = getCurrent();
    valDisplay.textContent = fmt(v);
    onChange(v);
  });

  // init display without firing onChange
  valDisplay.textContent = fmt(getCurrent());

  return {
    label: el("div", { class: "mlpt-slider-label" }, label),
    slider: sliderEl,
    display: valDisplay,
    getValue: getCurrent,
    setValue(v) {
      sliderEl.value = String(isLog ? valueToLogSlider(v, min, max) : (v - min) / (max - min));
      valDisplay.textContent = fmt(v);
    },
  };
}

/* ============================================================
   Main mount function
   ============================================================ */
export function mountMLPTrainer(root) {
  injectStyles();
  root.className = "mlpt-root";

  /* --- State --- */
  let running = false, paused = false, abortFlag = false;
  const control = { isPaused: () => paused, isAborted: () => abortFlag };

  let rawTrainArr = [], rawTestArr = [], stepArr = [];

  const grabList = [];
  let grabCounter = 0;
  const grabPlotMap = new Map();

  /* --- Config --- */
  const cfg = {
    d: 4,
    eigvals: [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.0001, 0.0001],
    coefA: 1.0, orderA: 1, dimA: 0,
    coefB: 0.0, orderB: 2, dimB: 1,
    lr: 1e-3, gamma: 1.0,
    width: 256, depth: 2, bsz: 256,
    activation: "relu",
    ema: 0.9,
  };

  /* ---- Status lines (no badge — just inline mono text) ---- */
  const errLine  = el("div", { class: "mlpt-errline" });
  const stepCtr  = el("div", { class: "mlpt-stepctr" });

  /* ---- TARGET FUNCTION DISPLAY ---- */
  const targetDisplay = el("div", { class: "mlpt-target-box" });

  function updateTargetDisplay() {
    const d  = cfg.d;
    const A  = cfg.coefA, oA = cfg.orderA, dA = Math.min(cfg.dimA, d - 1);
    const B  = cfg.coefB, oB = cfg.orderB, dB = Math.min(cfg.dimB, d - 1);

    const parts = [];
    parts.push(`<span class="hlA">${fmt(A, 3)}</span>·h<sub>${oA}</sub>(x<sub>${dA}</sub>)`);
    if (B !== 0) parts.push(`<span class="hlB">${fmt(B, 3)}</span>·h<sub>${oB}</sub>(x<sub>${dB}</sub>)`);
    const yExpr = parts.join(" + ");

    const eigStr = Array.from({ length: d }, (_, i) => {
      const v = cfg.eigvals[i] ?? 1;
      return `γ<sub>${i}</sub>=${fmt(v, 2)}`;
    }).join(", ");

    targetDisplay.innerHTML =
      `<span class="dim">x ~ N(0,Γ),   Γ=diag(${eigStr})</span><br>` +
      `y* = ${yExpr}`;
  }

  /* ---- EIGVAL SLIDERS ---- */
  const eigvalGrid = el("div", { class: "mlpt-eigval-grid" });

  function rebuildEigSliders(d) {
    eigvalGrid.innerHTML = "";
    const cols = Math.min(d, 5);
    eigvalGrid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    for (let i = 0; i < d; i++) {
      const idx = i;
      const startVal = Math.min(1, Math.max(1e-4, cfg.eigvals[idx] ?? 1.0));
      cfg.eigvals[idx] = startVal;
      const initRaw = valueToLogSlider(startVal, 1e-4, 1);

      const topLabel = el("div", {
        style: "font-family:'IBM Plex Mono',monospace;font-size:0.68rem;opacity:0.45;text-align:center;"
      }, `γ${idx}`);

      const sliderEl = el("input", {
        type: "range", class: "mlpt-range acc2",
        min: "0", max: "1", step: "any",
        value: String(initRaw), style: "width:100%;",
      });

      const numLabel = el("div", {
        style: "font-family:'IBM Plex Mono',monospace;font-size:0.68rem;text-align:center;opacity:0.85;"
      }, fmt(startVal, 2));

      sliderEl.addEventListener("input", () => {
        const v = logSliderValue(parseFloat(sliderEl.value), 1e-4, 1);
        cfg.eigvals[idx] = v;
        numLabel.textContent = fmt(v, 2);
        updateTargetDisplay();
      });

      eigvalGrid.append(
        el("div", { style: "display:flex;flex-direction:column;gap:3px;align-items:stretch;" },
          topLabel, sliderEl, numLabel)
      );
    }
  }

  /* ---- Helper: integer slider ---- */
  function makeIntSlider(label, initVal, min, max, accent, onSet) {
    const sl = makeSlider({
      label, min, max, value: initVal, isLog: false, accent,
      onChange(v) {
        const iv = Math.round(v);
        sl.display.textContent = String(iv);
        onSet(iv);
      }
    });
    // Force integer display on init (getCurrent() may be e.g. 3.9999)
    sl.display.textContent = String(initVal);
    return sl;
  }

  /* ---- DIMENSION ---- */
  const dimSl = makeIntSlider("d  (dimension)", cfg.d, 1, 10, "", (v) => {
    cfg.d = v;
    rebuildEigSliders(v);
    updateTargetDisplay();
  });

  /* ---- ARCHITECTURE ---- */
  const widthSl = makeSlider({
    label: "width", min: 128, max: 4096, value: cfg.width, isLog: true,
    onChange(v) { cfg.width = Math.round(v); widthSl.display.textContent = String(cfg.width); }
  });
  widthSl.display.textContent = String(cfg.width);

  const depthSl = makeIntSlider("depth", cfg.depth, 1, 4, "", (v) => { cfg.depth = v; });

  const actSel = el("select", { class: "mlpt-select" },
    ...["relu", "gelu", "tanh", "elu", "sigmoid", "linear"].map(a => el("option", { value: a }, a))
  );
  actSel.value = cfg.activation;
  actSel.addEventListener("change", () => { cfg.activation = actSel.value; });

  /* ---- OPTIMIZER ---- */
  const lrSl = makeSlider({
    label: "lr", min: 1e-3, max: 1, value: cfg.lr, isLog: true,
    onChange(v) { cfg.lr = v; }
  });
  const gammaSl = makeSlider({
    label: "γ  (gamma)", min: 1e-3, max: 1e3, value: cfg.gamma, isLog: true,
    onChange(v) { cfg.gamma = v; }
  });
  const bszSl = makeSlider({
    label: "batch size", min: 1, max: 2048, value: cfg.bsz, isLog: true,
    onChange(v) { cfg.bsz = Math.round(v); bszSl.display.textContent = String(cfg.bsz); }
  });
  bszSl.display.textContent = String(cfg.bsz);

  const optSel = el("select", { class: "mlpt-select" },
    el("option", { value: "adam" }, "adam"),
    el("option", { value: "sgd" },  "sgd")
  );

  /* ---- TARGET COEFFICIENTS / ORDERS / DIMS ---- */
  const coefASl = makeSlider({
    label: "α  (coef)", min: -3, max: 3, value: cfg.coefA, isLog: false,
    onChange(v) { cfg.coefA = v; updateTargetDisplay(); }
  });
  const coefBSl = makeSlider({
    label: "β  (coef)", min: -3, max: 3, value: cfg.coefB, isLog: false, accent: "acc2",
    onChange(v) { cfg.coefB = v; updateTargetDisplay(); }
  });

  const oASl = makeIntSlider("order",   cfg.orderA, 0, 6, "",     (v) => { cfg.orderA = v; updateTargetDisplay(); });
  const dASl = makeIntSlider("dim x_i", cfg.dimA,   0, 9, "",     (v) => { cfg.dimA   = v; updateTargetDisplay(); });
  const oBSl = makeIntSlider("order",   cfg.orderB, 0, 6, "acc2", (v) => { cfg.orderB = v; updateTargetDisplay(); });
  const dBSl = makeIntSlider("dim x_i", cfg.dimB,   0, 9, "acc2", (v) => { cfg.dimB   = v; updateTargetDisplay(); });

  /* ---- EMA DISPLAY SLIDER ---- */
  const emaSl = makeSlider({
    label: "EMA  (display)", min: 0, max: 0.999, value: cfg.ema, isLog: false, accent: "acc3",
    onChange(v) { cfg.ema = v; reRenderPlots(); }
  });

  /* ---- PLOTS (larger: 280px tall) ---- */
  const DPR = window.devicePixelRatio || 1;
  function makeCanvas(h = 280) {
    const c = el("canvas", { class: "mlpt-canvas" });
    c.width  = 1400 * DPR;
    c.height = h    * DPR;
    c.style.height = h + "px";
    return c;
  }
  const canvasTr = makeCanvas(280);
  const canvasTe = makeCanvas(280);

  const grabPlotsWrap = el("div", { style: "display:grid;gap:10px;margin-top:10px;" });

  function ensureGrabPlot(id, label) {
    if (grabPlotMap.has(id)) return grabPlotMap.get(id);
    const canvas = makeCanvas(200);
    const block = el("div", {},
      el("div", { style: "font-size:0.72rem;opacity:0.45;font-family:'IBM Plex Mono',monospace;margin-bottom:3px;" }, label),
      canvas
    );
    grabPlotsWrap.append(block);
    const obj = { canvas, ys: [], xs: [] };
    grabPlotMap.set(id, obj);
    return obj;
  }

  function reRenderPlots() {
    drawPlot(canvasTr, applyEma(rawTrainArr, cfg.ema), stepArr, "train loss", "#7c6af5");
    drawPlot(canvasTe, applyEma(rawTestArr,  cfg.ema), stepArr, "test loss",  "#4ec9a8");
  }

  /* ---- GRAB PANEL ---- */
  const grabsLatestOut  = el("div");
  const grabsListOut    = el("div");

  const grabTargetSel   = el("select", { class: "mlpt-select" }, ...GRAB_TARGETS.map(t => el("option", { value: t }, t)));
  const grabSummarySel  = el("select", { class: "mlpt-select" });
  const grabLayerInp    = el("input",  { class: "mlpt-numbox", type: "number", value: "0",  min: "0" });
  const grabKInp        = el("input",  { class: "mlpt-numbox", type: "number", value: "5",  min: "1", max: "32" });
  const grabCenteredSel = el("select", { class: "mlpt-select" },
    el("option", { value: "false" }, "raw"),
    el("option", { value: "true"  }, "centered")
  );
  const grabEveryInp = el("input", { class: "mlpt-numbox", type: "number", value: "50",  min: "10" });
  const probeSizeInp = el("input", { class: "mlpt-numbox", type: "number", value: "64",  min: "8", max: "128" });

  repopulateSummaries(grabTargetSel, grabSummarySel);
  grabTargetSel.addEventListener("change", () => repopulateSummaries(grabTargetSel, grabSummarySel));

  function refreshGrabsUI() {
    grabsListOut.innerHTML = "";
    if (!grabList.length) {
      grabsListOut.append(el("div", { style: "font-size:0.77rem;opacity:0.38;margin-top:4px;" }, "No grabs added."));
      return;
    }
    const table = el("table", { class: "mlpt-grab-table" });
    for (const g of grabList) {
      const desc = `${g.id} | ${g.target}:${g.summary}` +
        (g.layer !== undefined ? ` L${g.layer}` : "") +
        (g.k     !== undefined ? ` k=${g.k}`    : "");
      const rm = el("button", {
        class: "mlpt-btn mlpt-btn-danger",
        style: "padding:2px 9px;font-size:0.72rem;",
        onclick() {
          const i = grabList.findIndex(x => x.id === g.id);
          if (i >= 0) grabList.splice(i, 1);
          refreshGrabsUI();
        }
      }, "×");
      table.append(el("tr", {}, el("td", {}, desc), el("td", {}, rm)));
    }
    grabsListOut.append(table);
  }

  const addGrabBtn = el("button", {
    class: "mlpt-btn mlpt-btn-secondary",
    onclick() {
      const target   = grabTargetSel.value;
      const summary  = grabSummarySel.value;
      const layer    = Number(grabLayerInp.value);
      const k        = Number(grabKInp.value);
      const centered = grabCenteredSel.value === "true";
      const g = { id: `g${grabCounter++}`, target, summary };
      if (target !== "gradient") {
        g.layer = Number.isFinite(layer) ? layer : 0;
        if (summary.includes("topk")) g.k = Number.isFinite(k) ? k : 5;
        if (target === "activation") g.centered = centered;
      }
      grabList.push(g);
      refreshGrabsUI();
    }
  }, "+ Add");

  const clearGrabsBtn = el("button", {
    class: "mlpt-btn mlpt-btn-secondary",
    onclick() { grabList.splice(0, grabList.length); refreshGrabsUI(); }
  }, "Clear");

  /* ---- BUTTONS ---- */
  const pauseBtn = el("button", { class: "mlpt-btn mlpt-btn-secondary" }, "Pause");
  const resetBtn = el("button", { class: "mlpt-btn mlpt-btn-danger"    }, "Reset");

  const runBtn = el("button", {
    class: "mlpt-btn mlpt-btn-primary",
    async onclick() {
      if (running) return;
      running = true; abortFlag = false; paused = false;
      pauseBtn.textContent = "Pause";

      rawTrainArr = []; rawTestArr = []; stepArr = [];
      errLine.textContent  = "";
      stepCtr.textContent  = "";
      grabsLatestOut.innerHTML = "";
      grabPlotsWrap.innerHTML  = "";
      grabPlotMap.clear();
      reRenderPlots();

      const trainConfig = {
        data: {
          nTrain: 4000, nTest: 1000, seed: 42,
          dIn: cfg.d,
          eigvals: Array.from({ length: cfg.d }, (_, i) => cfg.eigvals[i] ?? 1.0),
          coefA: cfg.coefA, orderA: cfg.orderA, dimA: Math.min(cfg.dimA, cfg.d - 1),
          coefB: cfg.coefB, orderB: cfg.orderB, dimB: Math.min(cfg.dimB, cfg.d - 1),
          labelNoiseStd: 0.0,
        },
        model: {
          width: cfg.width, depth: cfg.depth,
          activation: cfg.activation, initScale: 1.0,
        },
        train: {
          lr: cfg.lr, gamma: cfg.gamma,
          optimizer: optSel.value,
          batchSize: cfg.bsz,
          maxIter: 100000,
          emaSmoother: 0.9,
          lossThreshold: 0,     // no threshold exit
          grabs: [...grabList],
          grabEvery: Number(grabEveryInp.value),
          probeSize: Number(probeSizeInp.value),
          control,
        },
      };

      try {
        await trainMLP(trainConfig, (p) => {
          if (p.paused) { stepCtr.textContent = `paused @ step ${p.step}`; return; }
          if (p.grabError) errLine.textContent = p.grabError;

          rawTrainArr.push(p.rawTrain);
          rawTestArr.push(p.rawTest);
          stepArr.push(p.step);

          stepCtr.textContent =
            `step ${p.step}  ·  tr=${p.emaTrain.toFixed(4)}  ·  te=${p.emaTest.toFixed(4)}`;

          reRenderPlots();

          if (p.lastGrabResults) {
            renderGrabLatest(grabsLatestOut, p.lastGrabResults);
            for (const [id, val] of Object.entries(p.lastGrabResults)) {
              const g = grabList.find(x => x.id === id);
              const label = g
                ? `${g.target}:${g.summary}${g.layer !== undefined ? `:L${g.layer}` : ""}`
                : id;
              const plot = ensureGrabPlot(id, label);
              plot.ys.push(scalarize(val));
              plot.xs.push(p.step);
              drawPlot(plot.canvas, plot.ys, plot.xs, label, "#f0a060");
            }
          }
        });
        stepCtr.textContent = abortFlag ? "reset." : "done.";
      } catch (e) {
        console.error(e);
        errLine.textContent = e?.message ?? String(e);
        stepCtr.textContent = "error.";
      } finally {
        running = false;
      }
    }
  }, "▶  Train");

  pauseBtn.addEventListener("click", () => {
    if (!running) return;
    paused = !paused;
    pauseBtn.textContent = paused ? "Resume" : "Pause";
  });

  resetBtn.addEventListener("click", () => {
    abortFlag = true; paused = false;
    pauseBtn.textContent = "Pause";
    errLine.textContent = "";
    stepCtr.textContent = "";
    rawTrainArr = []; rawTestArr = []; stepArr = [];
    grabsLatestOut.innerHTML = "";
    grabPlotsWrap.innerHTML  = "";
    grabPlotMap.clear();
    reRenderPlots();
  });

  /* ============================================================
     Layout helpers
     ============================================================ */
  function sliderRows(...sliders) {
    const grid = el("div", { class: "mlpt-slider-grid" });
    for (const sl of sliders) grid.append(sl.label, sl.slider, sl.display);
    return grid;
  }

  /* ============================================================
     Assemble layout
     ============================================================ */
  const targetCard = el("div", { class: "mlpt-card" },
    el("div", { class: "mlpt-card-title" }, "Target Function"),
    sliderRows(dimSl),
    el("div", { class: "mlpt-sep" }),
    el("div", {
      style: "font-size:0.68rem;opacity:0.4;margin-bottom:7px;font-family:'IBM Plex Mono',monospace;"
    }, "Data covariance eigenvalues  γ_i  (log scale)"),
    eigvalGrid,
    el("div", { class: "mlpt-sep" }),
    el("div", { style: "display:grid;grid-template-columns:1fr 1fr;gap:0 22px;" },
      el("div", {},
        el("div", { style: "font-size:0.68rem;color:#7c6af5;opacity:0.85;margin-bottom:6px;font-family:'IBM Plex Mono',monospace;" }, "Term A"),
        sliderRows(coefASl, oASl, dASl)
      ),
      el("div", {},
        el("div", { style: "font-size:0.68rem;color:#4ec9a8;opacity:0.85;margin-bottom:6px;font-family:'IBM Plex Mono',monospace;" }, "Term B"),
        sliderRows(coefBSl, oBSl, dBSl)
      )
    ),
    el("div", { class: "mlpt-sep" }),
    targetDisplay
  );

  const archCard = el("div", { class: "mlpt-card" },
    el("div", { class: "mlpt-card-title" }, "Architecture"),
    sliderRows(widthSl, depthSl),
    el("div", { style: "display:flex;align-items:center;gap:10px;margin-top:10px;" },
      el("div", { class: "mlpt-slider-label" }, "activation"),
      actSel
    )
  );

  const optCard = el("div", { class: "mlpt-card" },
    el("div", { class: "mlpt-card-title" }, "Optimizer"),
    sliderRows(lrSl, gammaSl, bszSl),
    el("div", { style: "display:flex;align-items:center;gap:10px;margin-top:10px;" },
      el("div", { class: "mlpt-slider-label" }, "optimizer"),
      optSel
    )
  );

  const trainCard = el("div", { class: "mlpt-card" },
    el("div", { class: "mlpt-card-title" }, "Training"),
    sliderRows(emaSl),
    el("div", { style: "display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;" },
      runBtn, pauseBtn, resetBtn
    ),
    el("div", { style: "margin-top:8px;" }, stepCtr, errLine)
  );

  const plotCard = el("div", { class: "mlpt-card" },
    el("div", { class: "mlpt-card-title" }, "Loss Curves"),
    el("div", { style: "display:grid;gap:10px;" }, canvasTr, canvasTe),
    grabPlotsWrap
  );

  const grabCard = el("div", { class: "mlpt-card" },
    el("div", { class: "mlpt-card-title" }, "Instrument Panel"),
    el("div", { class: "mlpt-grab-row" },
      el("span", { style: "font-size:0.76rem;opacity:0.45;" }, "target"),   grabTargetSel,
      el("span", { style: "font-size:0.76rem;opacity:0.45;" }, "summary"),  grabSummarySel,
      el("span", { style: "font-size:0.76rem;opacity:0.45;" }, "layer"),    grabLayerInp,
      el("span", { style: "font-size:0.76rem;opacity:0.45;" }, "k"),        grabKInp,
      el("span", { style: "font-size:0.76rem;opacity:0.45;" }, "centered"), grabCenteredSel,
      addGrabBtn, clearGrabsBtn
    ),
    el("div", { style: "display:flex;align-items:center;gap:10px;margin-top:8px;" },
      el("span", { style: "font-size:0.76rem;opacity:0.45;" }, "grab every"), grabEveryInp,
      el("span", { style: "font-size:0.76rem;opacity:0.45;" }, "probe size"), probeSizeInp
    ),
    el("div", { style: "margin-top:8px;" }, grabsListOut),
    el("div", { style: "margin-top:10px;font-size:0.68rem;opacity:0.38;font-family:'IBM Plex Mono',monospace;margin-bottom:5px;" },
      "Latest values"),
    grabsLatestOut
  );

  /* ---- Final assembly (no header) ---- */
  root.innerHTML = "";
  root.append(
    targetCard,
    el("div", { class: "mlpt-2col" }, archCard, optCard),
    trainCard,
    plotCard,
    grabCard
  );

  // Init
  rebuildEigSliders(cfg.d);
  updateTargetDisplay();
  refreshGrabsUI();
  reRenderPlots();
}
