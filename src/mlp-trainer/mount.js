// src/mlp-trainer/mount.js
import { trainMLP, GRAB_TARGETS } from "./trainer.js";

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "style") n.setAttribute("style", v);
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, String(v));
  }
  for (const kid of kids) n.append(kid);
  return n;
}

// Plot with labeled axes + autoscale + log-x (via log1p so step=0 is valid)
function drawPlot(canvas, ys, title, yLabel = "value", xs = null) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width,
    h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.font = "12px system-ui, sans-serif";

  if (!ys || ys.length < 2) {
    ctx.fillText(title, 10, 16);
    return;
  }

  const maxPoints = 900;
  const stride = Math.max(1, Math.floor(ys.length / maxPoints));

  const pts = [];
  const xpts = [];
  for (let i = 0; i < ys.length; i += stride) {
    pts.push(ys[i]);
    xpts.push(xs ? xs[i] : i);
  }

  // y-range
  let ymin = Infinity,
    ymax = -Infinity;
  for (const v of pts) {
    ymin = Math.min(ymin, v);
    ymax = Math.max(ymax, v);
  }
  if (!(ymax > ymin)) ymax = ymin + 1e-6;

  // padding: extra bottom room for ticks + xlabel (prevents overlap)
  const padL = 56,
    padB = 46,
    padT = 34,
    padR = 14;
  const left = padL,
    right = w - padR,
    top = padT,
    bottom = h - padB;

  // Title aligned to plot area (prevents overlap with y ticks)
  ctx.fillText(title, left + 4, 16);

  // Axes
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();

  // y tick labels
  ctx.fillText(ymax.toFixed(3), 6, top + 4);
  ctx.fillText(ymin.toFixed(3), 6, bottom + 4);

  // y-axis label (rotated)
  ctx.save();
  ctx.translate(16, (top + bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();

  // ---- log-x mapping (log1p) ----
  const logx = (v) => Math.log1p(Math.max(0, v)); // handles step=0 safely
  let xMin = Infinity,
    xMax = -Infinity;
  for (const xv of xpts) {
    xMin = Math.min(xMin, xv);
    xMax = Math.max(xMax, xv);
  }
  const lxMin = logx(xMin);
  const lxMax = logx(xMax);
  const lxDen = lxMax - lxMin > 0 ? lxMax - lxMin : 1e-9;

  const xToPix = (x) => left + ((logx(x) - lxMin) / lxDen) * (right - left);

  // ---- x ticks + xlabel on separate lines ----
  const tickY = h - 26; // tick numbers line
  const labelY = h - 8; // xlabel line

  const midL = 0.5 * (lxMin + lxMax);
  const xMid = Math.expm1(midL); // inverse of log1p

  const ticks = [
    { x: xMin, label: String(Math.round(xMin)) },
    { x: xMid, label: String(Math.round(xMid)) },
    { x: xMax, label: String(Math.round(xMax)) },
  ];

  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "#000";

  // draw ticks (with alignment so text doesn't run off canvas)
  for (let i = 0; i < ticks.length; i++) {
    const xp = xToPix(ticks[i].x);
    const lbl = ticks[i].label;
    if (i === 0) ctx.textAlign = "left";
    else if (i === ticks.length - 1) ctx.textAlign = "right";
    else ctx.textAlign = "center";
    ctx.fillText(lbl, xp, tickY);
  }

  // xlabel centered (won't collide with right tick)
  ctx.textAlign = "center";
  ctx.fillText("step (log)", 0.5 * (left + right), labelY);

  // ---- curve ----
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const x = xToPix(xpts[i]);
    const y = bottom - ((pts[i] - ymin) / (ymax - ymin)) * (bottom - top);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // restore default alignment
  ctx.textAlign = "start";
}

function fmtValue(v) {
  if (Array.isArray(v)) return `[${v.map((x) => Number(x).toFixed(4)).join(", ")}]`;
  if (typeof v === "number") return Number(v).toFixed(6);
  return String(v);
}

function renderGrabLatest(container, latest) {
  container.innerHTML = "";
  if (!latest) return;
  const table = el("table", { style: "width:100%; border-collapse:collapse; font-size:0.9rem;" });
  for (const [k, v] of Object.entries(latest)) {
    const row = el(
      "tr",
      {},
      el(
        "td",
        { style: "border-bottom:1px solid #eee; padding:4px 6px; width:40%; opacity:0.85;" },
        k
      ),
      el("td", { style: "border-bottom:1px solid #eee; padding:4px 6px;" }, fmtValue(v))
    );
    table.append(row);
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

// grab plotting: scalarize arrays by plotting first component
function scalarize(v) {
  if (typeof v === "number") return v;
  if (Array.isArray(v) && v.length) return Number(v[0]);
  return NaN;
}

// deterministic RNG for display previews
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// preview “first few terms”: not guaranteed to match trainer ordering perfectly,
// but gives an informative concrete expansion.
function enumerateFirstTerms({ d, cutoffMode, nTerms, maxDegree, cap = 8 }) {
  const terms = [];

  function genDegree(deg) {
    const out = [];
    function rec(dimStart, remaining, pairs) {
      if (remaining === 0) {
        out.push(pairs.slice());
        return;
      }
      for (let dim = dimStart; dim < d; dim++) {
        for (let k = 1; k <= remaining; k++) {
          pairs.push([dim, k]);
          rec(dim + 1, remaining - k, pairs);
          pairs.pop();
        }
      }
    }
    rec(0, deg, []);
    return out;
  }

  const hardCap = Math.min(cap, cutoffMode === "n_terms" ? nTerms : cap);

  if (cutoffMode === "max_degree") {
    for (let deg = 0; deg <= maxDegree; deg++) {
      if (terms.length >= hardCap) break;
      if (deg === 0) terms.push([]);
      else {
        for (const t of genDegree(deg)) {
          terms.push(t);
          if (terms.length >= hardCap) break;
        }
      }
    }
  } else {
    let deg = 0;
    while (terms.length < hardCap && deg <= 12) {
      if (deg === 0) terms.push([]);
      else {
        for (const t of genDegree(deg)) {
          terms.push(t);
          if (terms.length >= hardCap) break;
        }
      }
      deg++;
    }
  }

  return terms;
}

function termToString(term) {
  if (!term || term.length === 0) return "1";
  return term.map(([dim, ord]) => `h_${ord}(x_${dim})`).join("");
}

export function mountMLPTrainer(root) {
  // ---- state ----
  let running = false;
  let paused = false;
  let abortFlag = false;

  const control = {
    isPaused: () => paused,
    isAborted: () => abortFlag,
  };

  // ---- header/status ----
  const status = el("div", { style: "margin: 0.25rem 0; font-weight:600;" }, "Idle.");
  const errline = el("div", { style: "margin: 0.25rem 0; color: #b00; font-size: 0.9rem;" });

  // ---- plots ----
  const canvasTr = el("canvas", {
    width: 900,
    height: 180,
    style: "width: 100%; border: 1px solid #ddd; border-radius: 8px;",
  });
  const canvasTe = el("canvas", {
    width: 900,
    height: 180,
    style: "width: 100%; border: 1px solid #ddd; border-radius: 8px;",
  });

  // grab plots container
  const grabPlotsWrap = el("div", { style: "margin-top: 0.75rem; display: grid; gap: 0.75rem;" });
  const grabPlotMap = new Map(); // id -> { canvas, ys, xs }

  function ensureGrabPlot(id, label) {
    if (grabPlotMap.has(id)) return grabPlotMap.get(id);

    const title = el("div", { style: "font-weight:600;" }, label);
    const canvas = el("canvas", {
      width: 900,
      height: 150,
      style: "width:100%; border:1px solid #ddd; border-radius:8px;",
    });

    const block = el("div", {}, title, canvas);
    grabPlotsWrap.append(block);

    const obj = { canvas, ys: [], xs: [] };
    grabPlotMap.set(id, obj);
    return obj;
  }

  // ---- instrument panel ----
  const grabList = [];
  let grabCounter = 0;

  const grabsLatestOut = el("div", { style: "margin-top: 0.5rem;" });
  const grabsListOut = el("div", { style: "margin-top: 0.5rem;" });

  const grabTargetSel = el("select", {}, ...GRAB_TARGETS.map((t) => el("option", { value: t }, t)));
  const grabSummarySel = el("select", {});
  const grabLayerInp = el("input", { type: "number", value: "0", min: "0", style: "width:80px;" });
  const grabKInp = el("input", { type: "number", value: "5", min: "1", max: "32", style: "width:80px;" });
  const grabCenteredSel = el(
    "select",
    {},
    el("option", { value: "false" }, "raw"),
    el("option", { value: "true" }, "centered")
  );

  repopulateSummaries(grabTargetSel, grabSummarySel);
  grabTargetSel.addEventListener("change", () => repopulateSummaries(grabTargetSel, grabSummarySel));

  const grabEveryInp = el("input", { type: "number", value: "50", min: "10", style: "width:90px;" });
  const probeSizeInp = el("input", { type: "number", value: "64", min: "8", max: "128", style: "width:90px;" });

  function refreshGrabsUI() {
    grabsListOut.innerHTML = "";
    if (!grabList.length) {
      grabsListOut.append(el("div", { style: "opacity:0.75; font-size:0.9rem;" }, "No grabs added."));
      return;
    }

    const table = el("table", { style: "width:100%; border-collapse:collapse; font-size:0.9rem;" });
    for (const g of grabList) {
      const desc =
        `${g.id} | ${g.target}:${g.summary}` +
        (g.layer !== undefined ? ` | L${g.layer}` : "") +
        (g.k !== undefined ? ` | k=${g.k}` : "") +
        (g.centered !== undefined ? ` | ${g.centered ? "centered" : "raw"}` : "");

      const rm = el(
        "button",
        {
          style: "padding:2px 8px; border-radius:10px; border:1px solid #ccc; cursor:pointer;",
          onclick: () => {
            const i = grabList.findIndex((x) => x.id === g.id);
            if (i >= 0) grabList.splice(i, 1);
            refreshGrabsUI();
          },
        },
        "remove"
      );

      table.append(
        el(
          "tr",
          {},
          el("td", { style: "border-bottom:1px solid #eee; padding:4px 6px; width:80%;" }, desc),
          el("td", { style: "border-bottom:1px solid #eee; padding:4px 6px; text-align:right;" }, rm)
        )
      );
    }
    grabsListOut.append(table);
  }

  const addGrabBtn = el(
    "button",
    {
      style: "padding: 0.35rem 0.6rem; border-radius: 10px; border: 1px solid #ccc; cursor: pointer;",
      onclick: () => {
        const target = grabTargetSel.value;
        const summary = grabSummarySel.value;

        const layer = Number(grabLayerInp.value);
        const k = Number(grabKInp.value);
        const centered = grabCenteredSel.value === "true";

        const g = { id: `g${grabCounter++}`, target, summary };

        if (target === "gradient") {
          // gradients ignore layer/k/centered unless per-layer summary
        } else {
          g.layer = Number.isFinite(layer) ? layer : 0;
          if (summary.includes("topk")) g.k = Number.isFinite(k) ? k : 5;
          if (target === "activation") g.centered = centered;
        }

        grabList.push(g);
        refreshGrabsUI();
      },
    },
    "Add"
  );

  const clearGrabsBtn = el(
    "button",
    {
      style: "padding: 0.35rem 0.6rem; border-radius: 10px; border: 1px solid #ccc; cursor: pointer;",
      onclick: () => {
        grabList.splice(0, grabList.length);
        refreshGrabsUI();
      },
    },
    "Clear"
  );

  const instrumentPanel = el(
    "div",
    { style: "margin-top: 0.75rem; padding: 0.75rem; border: 1px solid #eee; border-radius: 12px;" },
    el("div", { style: "font-weight: 700; margin-bottom: 0.25rem;" }, "Grabs (safe instrument panel)"),
    el(
      "div",
      { style: "opacity: 0.8; font-size: 0.9rem; margin-bottom: 0.5rem;" },
      "Choose built-in probes only. Arrays plot their first component."
    ),

    el(
      "div",
      { style: "display:flex; flex-wrap:wrap; gap:0.5rem; align-items:end;" },
      el("label", {}, "Target ", grabTargetSel),
      el("label", {}, "Summary ", grabSummarySel),
      el("label", {}, "Layer ", grabLayerInp),
      el("label", {}, "k ", grabKInp),
      el("label", {}, "Act center ", grabCenteredSel),
      addGrabBtn,
      clearGrabsBtn,
      el("span", { style: "flex: 1 1 auto;" }, ""),
      el("label", {}, "Grab every ", grabEveryInp),
      el("label", {}, "Probe ", probeSizeInp)
    ),

    el("div", { style: "margin-top:0.5rem;" }, grabsListOut),
    el("div", { style: "margin-top:0.75rem; font-weight:700;" }, "Latest grab values"),
    grabsLatestOut
  );

  // ---- compact form helpers ----
  function inpNum(id, val, step = "1", min = null) {
    const attrs = { id, type: "number", value: String(val), step: String(step) };
    if (min !== null) attrs.min = String(min);
    return el("input", attrs);
  }
  function inpSel(id, options, val) {
    const s = el("select", { id });
    for (const [v, name] of options) s.append(el("option", { value: v }, name));
    s.value = val;
    return s;
  }

  // ---- compact form layout (8 columns) ----
  const form = el(
    "div",
    { style: "display:grid; grid-template-columns: repeat(8, minmax(0,1fr)); gap: 0.35rem; align-items:start;" },

    // dataset
    el("label", {}, "d", inpNum("dIn", 64, "1", 1)),
    el("label", {}, "X dist", inpSel("dist", [["gaussian", "gaussian"], ["uniform", "uniform"]], "gaussian")),
    el("label", {}, "||x||=1", inpSel("normalizeX", [["false", "false"], ["true", "true"]], "false")),
    el("label", {}, "noise σ", inpNum("labelNoiseStd", 0.0, "0.01", 0)),

    el("label", {}, "α (X)", inpNum("alpha", 1.2, "0.05")),
    el("label", {}, "X offset", inpNum("dataOffset", 1.0, "0.1")),
    el("label", {}, "X cutoff", inpSel("cutoffModeX", [["none", "none"], ["hard", "hard"]], "none")),
    el("label", {}, "X dims", inpNum("nTermsX", 64, "1", 1)),

    // teacher
    el(
      "label",
      {},
      "term cutoff",
      inpSel("cutoffMode", [["n_terms", "n_terms"], ["max_degree", "max_degree"]], "n_terms")
    ),
    el("label", {}, "nTerms", inpNum("nTerms", 256, "1", 1)),
    el("label", {}, "maxDeg", inpNum("maxDegree", 4, "1", 0)),
    el("label", {}, "teacher", inpSel("targetType", [["powerlaw", "powerlaw"], ["n_hot", "n_hot"]], "powerlaw")),

    el("label", {}, "β (teacher)", inpNum("beta", 1.2, "0.05")),
    el("label", {}, "t offset", inpNum("targetOffset", 1.0, "0.1")),
    el("label", {}, "nHot", inpNum("nHot", 16, "1", 1)),
    el("label", {}, "signs", inpSel("randomSigns", [["true", "true"], ["false", "false"]], "true")),

    // model
    el("label", {}, "width", inpNum("width", 256, "1", 1)),
    el("label", {}, "depth", inpNum("depth", 2, "1", 1)),
    el(
      "label",
      {},
      "act",
      inpSel(
        "act",
        [["relu", "relu"], ["tanh", "tanh"], ["sigmoid", "sigmoid"], ["gelu", "gelu"], ["elu", "elu"], ["linear", "linear"]],
        "relu"
      )
    ),
    el("label", {}, "init", inpNum("initScale", 1.0, "0.1")),

    // optim
    el("label", {}, "opt", inpSel("opt", [["sgd", "sgd"], ["adam", "adam"]], "adam")),
    el("label", {}, "lr", inpNum("lr", 0.01, "0.0001")),
    el("label", {}, "γ", inpNum("gamma", 1.0, "0.1")),
    el("label", {}, "bs", inpNum("bsz", 256, "1", 1)),

    // stopping / smoothing
    el("label", {}, "EMA", inpNum("ema", 0.9, "0.01", 0)),
    el("label", {}, "loss thr", inpNum("thr", 0.1, "0.01", 0)),
    el("div", { style: "opacity:0.8; align-self:center;" }, "max steps = 100000"),
    el("div", {}, "")
  );

  // --- make labels stacked / compact ---
  for (const lab of form.querySelectorAll("label")) {
    lab.style.display = "flex";
    lab.style.flexDirection = "column";
    lab.style.gap = "2px";
    lab.style.fontSize = "12px";
    lab.style.lineHeight = "1.1";
  }
  for (const node of form.querySelectorAll("input, select")) {
    node.style.padding = "2px 6px";
    node.style.borderRadius = "8px";
    node.style.border = "1px solid #cfcfcf";
    node.style.fontSize = "12px";
    node.style.height = "28px";
  }

  // --- manual n_hot expression row (hidden unless teacher === n_hot) ---
  const hotExprTa = el("textarea", {
    id: "hotExpr",
    rows: "2",
    placeholder: "e.g. 0.6h_1(x_10)+0.48h_2(x_0)h_1(x_3)",
    style: "resize: vertical;",
  });
  const hotExprRow = el(
    "div",
    { style: "grid-column: 1 / -1;" },
    el("label", {}, "manual y* (for n_hot)", hotExprTa)
  );
  form.append(hotExprRow);

  for (const node of form.querySelectorAll("textarea")) {
    node.style.padding = "6px 8px";
    node.style.borderRadius = "10px";
    node.style.border = "1px solid #cfcfcf";
    node.style.fontSize = "12px";
    node.style.minHeight = "48px";
  }

  // --- target function display ---
  const targetPre = el("pre", {
    style:
      "margin:0; padding:10px 12px; border-radius:12px; border:1px solid #ddd; background:#f6f3ea; " +
      "font-size:12px; white-space:pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, " +
      "'Liberation Mono', 'Courier New', monospace;",
  });
  const targetBox = el(
    "div",
    { style: "grid-column: 1 / -1;" },
    el("div", { style: "font-weight:600; margin: 2px 0 6px 0;" }, "Target function"),
    targetPre
  );
  form.append(targetBox);

  function toggleHotExprRow() {
    const teacher = form.querySelector("#targetType").value;
    hotExprRow.style.display = teacher === "n_hot" ? "block" : "none";
  }

  function updateTargetBox() {
    const dIn = Number(form.querySelector("#dIn").value);
    const dist = form.querySelector("#dist").value;
    const normalizeX = form.querySelector("#normalizeX").value === "true";
    const noise = Number(form.querySelector("#labelNoiseStd").value);

    const alpha = Number(form.querySelector("#alpha").value);
    const xOff = Number(form.querySelector("#dataOffset").value);
    const xCut = form.querySelector("#cutoffModeX").value;
    const xDims = Number(form.querySelector("#nTermsX").value);

    const termCut = form.querySelector("#cutoffMode").value;
    const nTerms = Number(form.querySelector("#nTerms").value);
    const maxDeg = Number(form.querySelector("#maxDegree").value);

    const teacher = form.querySelector("#targetType").value;
    const beta = Number(form.querySelector("#beta").value);
    const tOff = Number(form.querySelector("#targetOffset").value);
    const nHot = Number(form.querySelector("#nHot").value);
    const signs = form.querySelector("#randomSigns").value === "true";

    const hotExpr = (form.querySelector("#hotExpr")?.value ?? "").trim();

    const xdesc = `x ~ ${dist}${normalizeX ? " (projected to ||x||=1)" : ""}`;
    const scaled = `x-features scaled by (i + ${xOff})^(-${alpha.toFixed(2)})${
      xCut !== "none" ? ` with X-cutoff=${xCut}` : ""
    }`;
    const basis = `Hermite basis cutoff: ${termCut} (nTerms=${nTerms}, maxDeg=${maxDeg}), X dims=${xDims}`;

    const ndesc = noise > 0 ? `label noise ε ~ N(0, ${noise}^2)` : "no label noise";

    // Choose headline + preview
    let headline = `Target: y = f*(x)${noise > 0 ? " + ε" : ""}`;
    let preview = "";

    if (teacher === "n_hot" && hotExpr.length > 0) {
      headline = `Target: y* = ${hotExpr}${noise > 0 ? " + ε" : ""}`;
      preview = `Teacher: user-defined sparse Hermite expression (coeffs used exactly).`;
    } else if (teacher === "powerlaw") {
      const terms = enumerateFirstTerms({
        d: dIn,
        cutoffMode: termCut,
        nTerms,
        maxDegree: maxDeg,
        cap: 8,
      });

      const rng = mulberry32(42);
      const pieces = [];
      for (let t = 0; t < terms.length; t++) {
        const mag = Math.pow(Math.max(1e-6, t + tOff), -beta);
        const sgn = signs ? (rng() < 0.5 ? -1 : 1) : 1;
        const c = sgn * mag;
        const sym = termToString(terms[t]);
        pieces.push(`${c.toFixed(3)}·${sym}`);
      }
      preview = `First terms (powerlaw): ${pieces.join(" + ")}${signs ? " (signs deterministic for display)" : ""}`;
    } else {
      // n_hot but no manual expression entered
      preview = `Teacher: random n_hot=${nHot} sparse teacher${signs ? " with random ± signs" : ""}.`;
    }

    targetPre.textContent = `${headline}
${xdesc}
f*(x): Hermite-feature teacher on R^${dIn}
  - ${basis}
  - ${scaled}
  - ${preview}
  - ${ndesc}`;
  }

  // Keep target box in sync
  for (const node of form.querySelectorAll("input, select, textarea")) {
    node.addEventListener("input", updateTargetBox);
    node.addEventListener("change", updateTargetBox);
  }
  form.querySelector("#targetType").addEventListener("change", () => {
    toggleHotExprRow();
    updateTargetBox();
  });

  toggleHotExprRow();
  updateTargetBox();

  refreshGrabsUI();

  // ---- buttons: train/pause/reset ----
  const runBtn = el(
    "button",
    {
      style: "padding: 0.45rem 0.75rem; border-radius: 10px; border: 1px solid #ccc; cursor: pointer;",
      onclick: async () => {
        if (running) return;
        running = true;
        abortFlag = false;
        paused = false;
        pauseBtn.textContent = "Pause";

        errline.textContent = "";
        status.textContent = "Starting…";
        grabsLatestOut.innerHTML = "";

        // reset plots
        grabPlotsWrap.innerHTML = "";
        grabPlotMap.clear();

        const trainCurve = [];
        const testCurve = [];
        const trainSteps = [];
        const testSteps = [];
        drawPlot(canvasTr, [0, 0], "EMA train loss (MSE)", "loss");
        drawPlot(canvasTe, [0, 0], "EMA test loss (MSE)", "loss");

        const cfg = {
          data: {
            kind: "synthetic",
            nTrain: 4000,
            nTest: 1000,
            seed: 42,

            dIn: Number(root.querySelector("#dIn").value),
            distribution: root.querySelector("#dist").value,
            normalizeX: root.querySelector("#normalizeX").value === "true",

            cutoffModeX: root.querySelector("#cutoffModeX").value,
            nTermsX: Number(root.querySelector("#nTermsX").value),

            dataOffset: Number(root.querySelector("#dataOffset").value),
            alpha: Number(root.querySelector("#alpha").value),

            cutoffMode: root.querySelector("#cutoffMode").value,
            nTerms: Number(root.querySelector("#nTerms").value),
            maxDegree: Number(root.querySelector("#maxDegree").value),

            targetType: root.querySelector("#targetType").value,
            nHot: Number(root.querySelector("#nHot").value),
            randomSigns: root.querySelector("#randomSigns").value === "true",

            targetOffset: Number(root.querySelector("#targetOffset").value),
            beta: Number(root.querySelector("#beta").value),

            // NEW: user expression for n_hot (trainer.js must consume this)
            hotExpr: root.querySelector("#hotExpr")?.value ?? "",

            labelNoiseStd: Number(root.querySelector("#labelNoiseStd").value),
            maxTermsCap: 2048,
          },
          model: {
            width: Number(root.querySelector("#width").value),
            depth: Number(root.querySelector("#depth").value),
            activation: root.querySelector("#act").value,
            initScale: Number(root.querySelector("#initScale").value),
          },
          train: {
            lr: Number(root.querySelector("#lr").value),
            gamma: Number(root.querySelector("#gamma").value),
            optimizer: root.querySelector("#opt").value,
            batchSize: Number(root.querySelector("#bsz").value),

            maxIter: 100000,
            emaSmoother: Number(root.querySelector("#ema").value),
            lossThreshold: Number(root.querySelector("#thr").value),

            grabs: [...grabList],
            grabEvery: Number(grabEveryInp.value),
            probeSize: Number(probeSizeInp.value),

            control,
          },
        };

        try {
          await trainMLP(cfg, (p) => {
            if (p.paused) {
              status.textContent = `Paused @ step ${p.step}`;
              return;
            }

            if (p.grabError) errline.textContent = p.grabError;

            const doneStr = p.done ? " | Done." : "";
            status.textContent = `step ${p.step} | ema(train)=${p.emaTrain.toFixed(
              4
            )} | ema(test)=${p.emaTest.toFixed(4)} | thr=${(p.lossThreshold ?? NaN).toFixed(
              4
            )}${doneStr}`;

            trainCurve.push(p.emaTrain);
            testCurve.push(p.emaTest);
            trainSteps.push(p.step);
            testSteps.push(p.step);

            drawPlot(canvasTr, trainCurve, "EMA train loss (MSE)", "loss", trainSteps);
            drawPlot(canvasTe, testCurve, "EMA test loss (MSE)", "loss", testSteps);

            if (p.lastGrabResults) {
              renderGrabLatest(grabsLatestOut, p.lastGrabResults);

              for (const [id, val] of Object.entries(p.lastGrabResults)) {
                const g = grabList.find((x) => x.id === id);
                const label = g
                  ? `${g.target}:${g.summary}${g.layer !== undefined ? `:L${g.layer}` : ""}${
                      g.k !== undefined ? `:k${g.k}` : ""
                    }`
                  : id;

                const plot = ensureGrabPlot(id, label);
                plot.ys.push(scalarize(val));
                plot.xs.push(p.step);
                drawPlot(plot.canvas, plot.ys, label, "grab", plot.xs);
              }
            }
          });

          status.textContent = abortFlag ? "Reset." : "Done.";
        } catch (e) {
          console.error(e);
          errline.textContent = e?.message ?? String(e);
          status.textContent = "Error.";
        } finally {
          running = false;
        }
      },
    },
    "Train"
  );

  const pauseBtn = el(
    "button",
    {
      style: "padding: 0.45rem 0.75rem; border-radius: 10px; border: 1px solid #ccc; cursor: pointer;",
      onclick: () => {
        if (!running) return;
        paused = !paused;
        pauseBtn.textContent = paused ? "Resume" : "Pause";
      },
    },
    "Pause"
  );

  const resetBtn = el(
    "button",
    {
      style: "padding: 0.45rem 0.75rem; border-radius: 10px; border: 1px solid #ccc; cursor: pointer;",
      onclick: () => {
        abortFlag = true;
        paused = false;
        pauseBtn.textContent = "Pause";

        errline.textContent = "";
        status.textContent = "Reset.";
        grabsLatestOut.innerHTML = "";

        grabPlotsWrap.innerHTML = "";
        grabPlotMap.clear();

        drawPlot(canvasTr, [0, 0], "EMA train loss (MSE)", "loss");
        drawPlot(canvasTe, [0, 0], "EMA test loss (MSE)", "loss");
      },
    },
    "Reset"
  );

  // ---- assemble page ----
  root.innerHTML = "";
  root.append(
    el(
      "div",
      { style: "display:flex; justify-content:space-between; align-items:baseline; gap:1rem;" },
      el("h3", { style: "margin: 0;" }, "MLP Trainer (Synthetic Hermite Teacher)"),
      status
    ),
    errline,
    form,
    el("div", { style: "display:flex; gap:0.5rem; margin-top:0.5rem; flex-wrap:wrap;" }, runBtn, pauseBtn, resetBtn),
    el("div", { style: "margin-top: 0.75rem; display: grid; gap: 0.6rem;" }, canvasTr, canvasTe),
    grabPlotsWrap,
    instrumentPanel
  );
}
