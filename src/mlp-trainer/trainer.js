// public/mlp-trainer/trainer.js
import * as tf from "@tensorflow/tfjs";

// Deterministic PRNG for JS-side randomness (permutations, sign flips, etc.)
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Simple integer seed stream for TFJS random ops
function makeSeedStream(seed) {
  let s = (seed >>> 0) || 1;
  return function nextSeed() {
    // LCG
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s;
  };
}

/** =========================================================
 *  Hermite-teacher synthetic dataset (regression)
 *  y(x) = sum_{alpha in A} w_alpha * prod_j He_{alpha_j}(x_j)
 *  ========================================================= */

/**
 * Probabilists' Hermite polynomials He_n:
 * He_0(x)=1, He_1(x)=x, He_{n+1}(x)=x He_n(x) - n He_{n-1}(x)
 */
function hermiteTable1D(x, maxOrder) {
  // x: [B] tensor
  return tf.tidy(() => {
    const B = x.shape[0];
    const table = new Array(maxOrder + 1);
    table[0] = tf.ones([B], "float32");
    if (maxOrder >= 1) table[1] = x;
    for (let n = 1; n < maxOrder; n++) {
      table[n + 1] = x.mul(table[n]).sub(table[n - 1].mul(tf.scalar(n, "float32")));
    }
    return table; // array length maxOrder+1
  });
}

/** Anisotropic scaling vector: scale_i = (i + offset)^(-exp) */
function makeScaleVec(d, offset, exp) {
  return tf.tidy(() => {
    const idx = tf.range(0, d, 1, "float32").add(tf.scalar(offset, "float32"));
    const safe = tf.maximum(idx, tf.scalar(1e-6));
    return safe.pow(tf.scalar(-exp, "float32")); // [d]
  });
}

/**
 * Enumerate multi-indices in graded-lex order by total degree.
 * Sparse storage: each term is array of [dim, order] with order>=1.
 *
 * cutoffMode:
 *  - "n_terms": take first nTerms terms
 *  - "max_degree": all terms with total degree <= maxDegree (capped)
 */
function enumerateMultiIndices({
  d,
  cutoffMode = "n_terms",
  nTerms = 256,
  maxDegree = 4,
  maxTermsCap = 2048, // safety cap
}) {
  const cap = Math.min(maxTermsCap, cutoffMode === "n_terms" ? nTerms : maxTermsCap);
  const terms = [];

  // generate all sparse terms of total degree = deg
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

  if (cutoffMode === "max_degree") {
    for (let deg = 0; deg <= maxDegree; deg++) {
      if (deg === 0) {
        terms.push([]); // constant term
      } else {
        const degTerms = genDegree(deg);
        for (const t of degTerms) {
          terms.push(t);
          if (terms.length >= cap) return terms;
        }
      }
    }
    return terms;
  }

  // cutoffMode === "n_terms"
  let deg = 0;
  while (terms.length < cap) {
    if (deg === 0) {
      terms.push([]);
    } else {
      const degTerms = genDegree(deg);
      for (const t of degTerms) {
        terms.push(t);
        if (terms.length >= cap) break;
      }
    }
    deg++;
    // prevent combinatorial blow-up from going too far
    if (deg > 12) break;
  }
  return terms;
}

/**
 * Teacher weights over terms:
 *  - powerlaw: w_t ∝ (t + targetOffset)^(-beta)
 *  - n_hot: choose nHot terms, apply same powerlaw magnitudes on those
 * Optionally random signs. Always L2-normalize weights.
 */
function makeTeacherWeights({
  nTerms,
  targetType = "powerlaw",
  beta = 0.0,
  targetOffset = 1.0,
  nHot = 16,
  randomSigns = true,
  jsRand,
}) {
  const mags = new Float32Array(nTerms);
  for (let t = 0; t < nTerms; t++) {
    const base = Math.max(1e-6, (t + targetOffset));
    mags[t] = Math.pow(base, -beta);
  }

  const w = new Float32Array(nTerms);

  if (targetType === "n_hot") {
    const k = Math.max(1, Math.min(Math.floor(nHot), nTerms));
    const perm = Array.from({ length: nTerms }, (_, i) => i);
    for (let i = nTerms - 1; i > 0; i--) {
      const j = Math.floor(jsRand() * (i + 1));
      [perm[i], perm[j]] = [perm[j], perm[i]];
    }
    const hot = new Set(perm.slice(0, k));
    for (let t = 0; t < nTerms; t++) if (hot.has(t)) w[t] = mags[t];
  } else {
    for (let t = 0; t < nTerms; t++) w[t] = mags[t];
  }

  if (randomSigns) {
    for (let t = 0; t < nTerms; t++) {
      if (w[t] !== 0) w[t] *= (jsRand() < 0.5 ? -1 : 1);
    }
  }

  // L2 normalize
  let ss = 0;
  for (let t = 0; t < nTerms; t++) ss += w[t] * w[t];
  const norm = Math.sqrt(Math.max(1e-12, ss));
  for (let t = 0; t < nTerms; t++) w[t] /= norm;

  return tf.tensor1d(w, "float32");
}

/**
 * Evaluate Hermite features Phi(X) in R^{B x T} for sparse terms.
 * X: [B, d]
 */
function evalHermiteFeatures(X, termsSparse, maxOrder) {
  return tf.tidy(() => {
    const [B, d] = X.shape;

    // Precompute He_n(x_j) for each dim j up to maxOrder
    const hermByDim = new Array(d);
    for (let j = 0; j < d; j++) {
      const xj = X.slice([0, j], [B, 1]).reshape([B]); // [B]
      hermByDim[j] = hermiteTable1D(xj, maxOrder);     // array length maxOrder+1
    }

    // Columns
    const cols = [];
    for (const term of termsSparse) {
      let col = tf.ones([B], "float32");
      for (const [dim, order] of term) {
        col = col.mul(hermByDim[dim][order]);
      }
      cols.push(col);
    }

    return tf.stack(cols, 0).transpose(); // [B, T]
  });
}

// Parse strings like: 0.6h_1(x_10)+0.48h_2(x_0)h_1(x_3)
function parseHotExpr(expr) {
  const s0 = String(expr ?? "").trim();
  if (!s0) return [];

  const s = s0.replace(/\s+/g, "").replace(/\*/g, "");
  const parts = s.match(/[+-]?[^+-]+/g);
  if (!parts) throw new Error(`Could not parse hotExpr: "${s0}"`);

  const out = [];
  for (const raw of parts) {
    if (!raw) continue;

    // coefficient prefix (optional)
    let coef = 1.0;
    let rest = raw;

    // If it begins with '+' or '-', keep that sign even when coef omitted
    const sign = rest[0] === "-" ? -1 : 1;
    if (rest[0] === "+" || rest[0] === "-") rest = rest.slice(1);

    // number like 1, 1.2, .3, 1e-2
    const m = rest.match(/^((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?/);
    if (m) {
      coef = Number(m[0]);
      rest = rest.slice(m[0].length);
    } else {
      coef = 1.0;
    }
    coef *= sign;

    // parse factors h_k(x_j)
    const factors = [];
    const re = /h_(\d+)\(x_(\d+)\)/g;
    let mm;
    let consumed = 0;
    while ((mm = re.exec(rest)) !== null) {
      factors.push([Number(mm[2]), Number(mm[1])]); // [dim, order]
      consumed = re.lastIndex;
    }
    if (rest.length !== consumed) {
      throw new Error(
        `Bad factor syntax in hotExpr term "${raw}". Expected factors like h_2(x_0)h_1(x_3).`
      );
    }
    // Reject repeated dims (our basis uses one Hermite per dim)
    const dims = new Set();
    for (const [dim] of factors) {
      if (dims.has(dim)) {
        throw new Error(`hotExpr term "${raw}" repeats x_${dim}. Use at most one h_k(x_dim) per dim.`);
      }
      dims.add(dim);
    }

    // canonicalize by dim
    factors.sort((a, b) => a[0] - b[0]);
    out.push({ coef, factors });
  }
  return out;
}

function termKey(factors) {
  // factors: array of [dim, order], sorted by dim
  if (!factors || factors.length === 0) return "";
  return factors.map(([d, k]) => `${d}:${k}`).join(",");
}

// Build a weight vector (Float32Array) aligned with termsSparse ordering.
// Coeffs are used EXACTLY as typed (no normalization, no random signs).
function weightsFromHotExpr({ termsSparse, hotExpr }) {
  const parsed = parseHotExpr(hotExpr);
  if (parsed.length === 0) return null;

  const keyToIndex = new Map();
  for (let i = 0; i < termsSparse.length; i++) {
    const key = termKey(termsSparse[i]);
    keyToIndex.set(key, i);
  }

  const w = new Float32Array(termsSparse.length);
  for (const { coef, factors } of parsed) {
    const key = termKey(factors);
    const idx = keyToIndex.get(key);
    if (idx === undefined) {
      throw new Error(
        `hotExpr refers to term "${key}" not present in current basis. ` +
          `Increase maxDeg / nTerms, or switch cutoffMode.`
      );
    }
    w[idx] += coef;
  }

  return tf.tensor1d(w, "float32");
}

/**
 * Synthetic dataset (regression labels y in R^{n x 1})
 *
 * X distribution:
 *  - gaussian: scale_i * N(0,1)
 *  - uniform:  scale_i * U[-sqrt(3), sqrt(3)]  (unit-variance uniform)
 *
 * X anisotropy:
 *  scale_i = (i + dataOffset)^(-alpha)
 *
 * X normalization (optional):
 *  normalizeX: rescale each sample so ||x||_2 = 1
 *
 * X cutoff (optional):
 *  cutoffModeX: "none" | "hard" (zero out dims >= nTermsX)
 *
 * Teacher terms:
 *  cutoffMode: "n_terms" | "max_degree"
 *  nTerms or maxDegree + maxTermsCap
 */
export function makeSyntheticDataset({
  nTrain = 4000,
  nTest = 1000,
  dIn = 64,

  distribution = "gaussian", // "gaussian" | "uniform"
  normalizeX = false,

  cutoffModeX = "none", // "none" | "hard"
  nTermsX = 64,

  dataOffset = 1.0,
  alpha = 0.0,

  // term set
  cutoffMode = "n_terms", // "n_terms" | "max_degree"
  nTerms = 256,
  maxDegree = 4,
  maxTermsCap = 2048,

  // teacher weights over terms
  targetType = "powerlaw", // "powerlaw" | "n_hot"
  beta = 0.0,
  targetOffset = 1.0,
  nHot = 16,
  randomSigns = true,
  hotExpr = "",

  // additive Gaussian label noise (std)
  labelNoiseStd = 0.0,

  seed = 42,
} = {}) {
  const d = Math.max(1, Math.floor(dIn));
  const nX = Math.max(1, Math.floor(nTermsX));
  const noiseStd = Math.max(0, Number(labelNoiseStd));

  const seedInt = (Number(seed) | 0) >>> 0;
  const jsRand = mulberry32(seedInt);
  const nextSeed = makeSeedStream(seedInt);


  // tf.util.seedrandom(String(seed));

  // enumerate terms + get max Hermite order needed
  const termsSparse = enumerateMultiIndices({ d, cutoffMode, nTerms, maxDegree, maxTermsCap });
  let maxOrder = 0;
  for (const term of termsSparse) for (const [, ord] of term) maxOrder = Math.max(maxOrder, ord);

  let w = null;

  // if n_hot + user supplied expression, use it (no normalization)
  if (targetType === "n_hot" && String(hotExpr).trim().length > 0) {
    w = weightsFromHotExpr({ termsSparse, hotExpr }); // you'll add this helper below
  }

  if (!w) {
    w = makeTeacherWeights({
      nTerms: termsSparse.length,
      targetType,
      beta,
      targetOffset,
      nHot,
      randomSigns,
      jsRand,
    });
  }


  function sampleX(n) {
    return tf.tidy(() => {
      const scale = makeScaleVec(d, dataOffset, alpha).reshape([1, d]);
      let X;
      if (distribution === "uniform") {
        const a = Math.sqrt(3);
        X = tf.randomUniform([n, d], -a, a, "float32", nextSeed());
      } else {
        X = tf.randomNormal([n, d], 0, 1, "float32", nextSeed());
      }
      X = X.mul(scale);

      if (cutoffModeX === "hard") {
        const m = Math.min(Math.max(1, nX), d);
        const mask = tf.concat(
          [tf.ones([m], "float32"), tf.zeros([Math.max(0, d - m)], "float32")],
          0
        ).reshape([1, d]);
        X = X.mul(mask);
      }

      if (normalizeX) {
        const eps = tf.scalar(1e-12, "float32");
        const norms = X.norm("euclidean", 1).add(eps).reshape([n, 1]);
        X = X.div(norms);
      }
      return X;
    });
  }

  function makeY(X) {
    return tf.tidy(() => {
      const Phi = evalHermiteFeatures(X, termsSparse, maxOrder);    // [B, T]
      let y = Phi.matMul(w.reshape([-1, 1]));                      // [B, 1]
      const yStd = tf.moments(y).variance.sqrt().dataSync()[0];
      console.log("teacher y std =", yStd);
      if (noiseStd > 0) y = y.add(tf.randomNormal([X.shape[0], 1], 0, noiseStd, "float32", nextSeed()));
      return y;
    });
  }

  const Xtr = sampleX(nTrain);
  const Xte = sampleX(nTest);
  const ytr = makeY(Xtr);
  const yte = makeY(Xte);
  w.dispose();

  return { train: { X: Xtr, y: ytr }, test: { X: Xte, y: yte }, dIn: d, numClasses: null };
}

/** =========================================================
 *  MLP model
 *  ========================================================= */

const ACTIVATIONS = new Set(["relu", "tanh", "sigmoid", "gelu", "elu", "linear"]);
function assertActivation(a) {
  if (!ACTIVATIONS.has(a)) throw new Error(`Unknown activation: ${a}`);
}

export function buildMLP({
  dIn,
  dOut = 1,
  width = 256,
  depth = 2, // number of hidden dense layers
  activation = "relu",
  initScale = 1.0,
}) {
  assertActivation(activation);

  const model = tf.sequential();

  function kernelInit(fanIn) {
    const std = initScale / Math.sqrt(Math.max(1, fanIn));
    return tf.initializers.randomNormal({ mean: 0, stddev: std });
  }

  model.add(
    tf.layers.dense({
      inputShape: [dIn],
      units: width,
      activation: activation === "linear" ? undefined : activation,
      kernelInitializer: kernelInit(dIn),
      biasInitializer: "zeros",
    })
  );

  for (let i = 1; i < depth; i++) {
    model.add(
      tf.layers.dense({
        units: width,
        activation: activation === "linear" ? undefined : activation,
        kernelInitializer: kernelInit(width),
        biasInitializer: "zeros",
      })
    );
  }

  model.add(
    tf.layers.dense({
      units: dOut,
      activation: undefined, // regression output
      kernelInitializer: kernelInit(width),
      biasInitializer: "zeros",
    })
  );

  return model;
}

/** =========================================================
 *  Safe instrument panel grabs (whitelist only)
 *  ========================================================= */

export const GRAB_TARGETS = ["weight", "activation", "gradient"];
export const GRAB_SUMMARIES = [
  "rms",
  "fro_norm",
  "spectral_norm",
  "trace_gram",
  "topk_singular_values",
  "topk_psd_eigenvalues",
  "grad_global_l2",
  "grad_per_layer_rms",
];

const CAPS = Object.freeze({
  grabEveryMin: 10,
  kMax: 32,
  probeMax: 128,
  exactSvdMaxDim: 512,
  randomizedIters: 1,
  powerIters: 12,
});

function clampInt(x, lo, hi) {
  x = Math.floor(Number(x));
  if (!Number.isFinite(x)) x = lo;
  return Math.max(lo, Math.min(hi, x));
}

function gammaRescaledLR(lr, gamma) {
  return gamma >= 1 ? lr * gamma : lr * (gamma * gamma);
}

function getDenseLayers(model) {
  return model.layers.filter((L) => L.getClassName && L.getClassName() === "Dense");
}

function getWeightMatrix(model, denseLayerIndex) {
  const denseLayers = getDenseLayers(model);
  if (denseLayerIndex < 0 || denseLayerIndex >= denseLayers.length) {
    throw new Error(`Weight grab: layer index ${denseLayerIndex} out of range.`);
  }
  // kernel only (ignore bias)
  const w = denseLayers[denseLayerIndex].getWeights()[0];
  return w;
}

function getActivationModelCached(cache, model, denseLayerIndex) {
  const key = `act_${denseLayerIndex}`;
  if (cache[key]) return cache[key];

  const denseLayers = getDenseLayers(model);
  if (denseLayerIndex < 0 || denseLayerIndex >= denseLayers.length) {
    throw new Error(`Activation grab: layer index ${denseLayerIndex} out of range.`);
  }
  const targetLayer = denseLayers[denseLayerIndex];
  const actModel = tf.model({ inputs: model.inputs, outputs: targetLayer.output });
  cache[key] = actModel;
  return actModel;
}

async function spectralNormPowerIter(W, iters = CAPS.powerIters) {
  return tf.tidy(async () => {
    const [m, n] = W.shape;
    let v = tf.randomNormal([n, 1]);
    v = v.div(v.norm().add(1e-12));
    for (let i = 0; i < iters; i++) {
      const u = W.matMul(v);
      const uN = u.div(u.norm().add(1e-12));
      const v2 = W.transpose().matMul(uN);
      v = v2.div(v2.norm().add(1e-12));
    }
    const sigma = W.matMul(v).norm();
    return (await sigma.data())[0];
  });
}

async function topKSingularValues(W, k) {
  const [m, n] = W.shape;
  const maxDim = Math.max(m, n);

  if (maxDim <= CAPS.exactSvdMaxDim) {
    const s = tf.linalg.svd(W, true).s;
    const sArr = Array.from(await s.data());
    sArr.sort((a, b) => b - a);
    return sArr.slice(0, k);
  }

  // randomized SVD
  if (!tf.linalg.qr) {
    throw new Error("Approx SVD requires tf.linalg.qr; not available in this TFJS build.");
  }

  const p = 5;
  const r = Math.min(n, k + p);

  const vals = await tf.tidy(async () => {
    let Y = W.matMul(tf.randomNormal([n, r]));
    for (let i = 0; i < CAPS.randomizedIters; i++) {
      Y = W.matMul(W.transpose().matMul(Y));
    }
    const qr = tf.linalg.qr(Y);
    const Q = qr.q;                 // m x r
    const B = Q.transpose().matMul(W); // r x n
    const sSmall = tf.linalg.svd(B, true).s;
    const sArr = Array.from(await sSmall.data());
    sArr.sort((a, b) => b - a);
    return sArr.slice(0, k);
  });

  return vals;
}

function centerBatch(H) {
  const mean = H.mean(0, true);
  return H.sub(mean);
}

async function computeWeightGrab(model, grab) {
  const layer = clampInt(grab.layer, 0, 1e9);
  const summary = grab.summary;
  const k = clampInt(grab.k ?? 5, 1, CAPS.kMax);

  const W = getWeightMatrix(model, layer);

  if (summary === "rms") {
    return tf.tidy(async () => (await W.square().mean().sqrt().data())[0]);
  }
  if (summary === "fro_norm") {
    return tf.tidy(async () => (await W.norm("fro").data())[0]);
  }
  if (summary === "spectral_norm") {
    return await spectralNormPowerIter(W);
  }
  if (summary === "trace_gram") {
    // tr(W^T W) = ||W||_F^2
    return tf.tidy(async () => (await W.square().sum().data())[0]);
  }
  if (summary === "topk_singular_values") {
    return await topKSingularValues(W, k);
  }
  if (summary === "topk_psd_eigenvalues") {
    const s = await topKSingularValues(W, k);
    return s.map((x) => x * x);
  }
  throw new Error(`Unsupported weight summary: ${summary}`);
}

async function computeActivationGrab(model, cache, grab, Xprobe) {
  const layer = clampInt(grab.layer, 0, 1e9);
  const summary = grab.summary;
  const k = clampInt(grab.k ?? 5, 1, CAPS.kMax);
  const centered = !!grab.centered;

  const actModel = getActivationModelCached(cache, model, layer);

  return tf.tidy(async () => {
    let H = actModel.apply(Xprobe, { training: false });
    if (!(H instanceof tf.Tensor)) throw new Error("Activation model returned non-tensor.");
    if (H.rank !== 2) H = H.reshape([H.shape[0], H.size / H.shape[0]]);
    if (centered) H = centerBatch(H);

    if (summary === "rms") {
      return (await H.square().mean().sqrt().data())[0];
    }
    if (summary === "trace_gram") {
      return (await H.square().sum().data())[0];
    }
    if (summary === "fro_norm") {
      return (await H.norm("fro").data())[0];
    }
    if (summary === "spectral_norm") {
      const s = tf.linalg.svd(H, true).s;
      return (await s.data())[0];
    }
    if (summary === "topk_singular_values") {
      const s = tf.linalg.svd(H, true).s;
      const sArr = Array.from(await s.data());
      sArr.sort((a, b) => b - a);
      return sArr.slice(0, k);
    }
    if (summary === "topk_psd_eigenvalues") {
      const s = tf.linalg.svd(H, true).s;
      const sArr = Array.from(await s.data());
      sArr.sort((a, b) => b - a);
      return sArr.slice(0, k).map((x) => x * x);
    }
    throw new Error(`Unsupported activation summary: ${summary}`);
  });
}

function mseLoss(logits, y) {
  const y2 = y.reshape([-1, 1]);
  return tf.losses.meanSquaredError(y2, logits).mean();
}

async function computeGradientGrab(model, grab, Xprobe, yprobe) {
  const summary = grab.summary;

  // differentiate w.r.t. trainable vars
  const vars = model.trainableWeights.map((w) => w.val);

  return tf.tidy(async () => {
    const { grads } = tf.variableGrads(() => {
      const logits = model.apply(Xprobe, { training: false });
      return mseLoss(logits, yprobe);
    }, vars);

    const gradTensors = vars.map((v) => grads[v.name]).filter(Boolean);

    if (summary === "rms") {
      // global RMS over all grad entries: sqrt(mean(g^2))
      const totalSq = tf.addN(gradTensors.map((g) => g.square().sum()));
      const totalN = gradTensors.reduce((acc, g) => acc + g.size, 0);
      const rms = totalSq.div(tf.scalar(Math.max(1, totalN))).sqrt();
      return (await rms.data())[0];
    }

    if (summary === "grad_global_l2") {
      const s = tf.addN(gradTensors.map((g) => g.square().sum()));
      return (await s.sqrt().data())[0];
    }

    if (summary === "grad_per_layer_rms") {
      // Dense kernel grads RMS per layer (best-effort)
      const denseLayers = getDenseLayers(model);
      const arr = [];
      for (let i = 0; i < denseLayers.length; i++) {
        const w = denseLayers[i].getWeights()[0];
        const g = grads[w.name];
        if (!g) {
          arr.push(NaN);
          continue;
        }
        arr.push((await g.square().mean().sqrt().data())[0]);
      }
      return arr;
    }

    throw new Error(`Unsupported gradient summary: ${summary}`);
  });
}


async function computeGrab(model, cache, grab, ctx) {
  if (!GRAB_TARGETS.includes(grab.target)) throw new Error(`Unknown grab target: ${grab.target}`);
  if (!GRAB_SUMMARIES.includes(grab.summary)) throw new Error(`Unknown grab summary: ${grab.summary}`);

  if (grab.target === "weight") return await computeWeightGrab(model, grab);
  if (grab.target === "activation") return await computeActivationGrab(model, cache, grab, ctx.Xprobe);
  if (grab.target === "gradient") return await computeGradientGrab(model, grab, ctx.Xprobe, ctx.yprobe);

  throw new Error(`Unhandled grab target: ${grab.target}`);
}

/** =========================================================
 *  Train loop (synthetic regression) + loss curves + grabs
 *  ========================================================= */

export async function trainMLP(config, onProgress = () => {}) {
  const { data, model: modelCfg, train: trainCfg } = config;

    const {
    // train
    maxIter = 100000,         // default, but UI will hard-set too
    batchSize = 256,
    lr = 1e-2,
    gamma = 1.0,
    optimizer = "sgd",        // "sgd" | "adam"

    // reporting
    emaSmoother = 0.9,
    lossThreshold = 0.1,      // single float

    // safe grabs
    grabs = [],
    grabEvery = 50,
    probeSize = 64,

    // control (optional)
    control = null,           // { isPaused:()=>bool, isAborted:()=>bool }
  } = trainCfg;

  const safeGrabEvery = clampInt(grabEvery, CAPS.grabEveryMin, 1e9);
  const safeProbe = clampInt(probeSize, 8, CAPS.probeMax);

  // dataset (synthetic only)
  const ds = makeSyntheticDataset({
    nTrain: data.nTrain ?? 4000,
    nTest: data.nTest ?? 1000,
    dIn: data.dIn ?? 64,
    distribution: data.distribution ?? "gaussian",
    normalizeX: data.normalizeX ?? false,
    cutoffModeX: data.cutoffModeX ?? "none",
    nTermsX: data.nTermsX ?? (data.dIn ?? 64),
    dataOffset: data.dataOffset ?? 1.0,
    alpha: data.alpha ?? 0.0,

    cutoffMode: data.cutoffMode ?? "n_terms",
    nTerms: data.nTerms ?? 256,
    maxDegree: data.maxDegree ?? 4,
    maxTermsCap: data.maxTermsCap ?? 2048,

    targetType: data.targetType ?? "powerlaw",
    beta: data.beta ?? 0.0,
    targetOffset: data.targetOffset ?? 1.0,
    nHot: data.nHot ?? 16,
    randomSigns: data.randomSigns ?? true,
    hotExpr: data.hotExpr ?? "",

    labelNoiseStd: data.labelNoiseStd ?? 0.0,
    seed: data.seed ?? 42,
  });

  const Xtr = ds.train.X;
  const ytr = ds.train.y; // [n,1]
  const Xte = ds.test.X;
  const yte = ds.test.y; // [m,1]
  const dIn = ds.dIn;

  // model (regression output dim = 1)
  const model = buildMLP({
    dIn,
    dOut: 1,
    width: modelCfg.width ?? 256,
    depth: modelCfg.depth ?? 2,
    activation: modelCfg.activation ?? "relu",
    initScale: modelCfg.initScale ?? 1.0,
  });

  // optimizer
  const effLR = gammaRescaledLR(lr, gamma);
  const opt = optimizer === "adam" ? tf.train.adam(effLR) : tf.train.sgd(effLR);

  // thresholds (same vibe as trainloop)
  const thr = Number(lossThreshold);
  const hasThr = Number.isFinite(thr) && thr > 0;
  let timekey = null; // first step where emaTrain < thr

  const trainCurve = [];
  const testCurve = [];

  // fixed probe batch for grabs
  const nTrain = Xtr.shape[0];
  const probeN = Math.min(safeProbe, nTrain);
  const Xprobe = Xtr.slice([0, 0], [probeN, dIn]);
  const yprobe = ytr.slice([0, 0], [probeN, 1]);

  const grabSeries = {}; // id -> [{step,value}]
  const grabCache = {};

  let emaTr = null;
  let emaTe = null;

  function batchIndices(step) {
    const start = (step * batchSize) % nTrain;
    const end = Math.min(nTrain, start + batchSize);
    return { start, end };
  }

  for (let step = 0; step < maxIter; step++) {
    // pause / abort controls
    if (control?.isAborted?.()) break;
    while (control?.isPaused?.()) {
      onProgress({ step, paused: true });
      await tf.nextFrame();
      if (control?.isAborted?.()) break;
    }
    if (control?.isAborted?.()) break;

    const { start, end } = batchIndices(step);

    const { trLossVal, teLossVal } = await tf.tidy(() => {
      const xb = Xtr.slice([start, 0], [end - start, dIn]);
      const yb = ytr.slice([start, 0], [end - start, 1]);

      const trLossTensor = opt.minimize(() => {
        const yhat = model.apply(xb, { training: true });
        return mseLoss(yhat, yb);
      }, true);

      const trLossVal = trLossTensor.dataSync()[0];

      const yhatTe = model.apply(Xte, { training: false });
      const teLossTensor = mseLoss(yhatTe, yte);
      const teLossVal = teLossTensor.dataSync()[0];

      return { trLossVal, teLossVal };
    });

    if (step === 0) {
      emaTr = trLossVal;
      emaTe = teLossVal;
    }
    emaTr = emaSmoother * emaTr + (1 - emaSmoother) * trLossVal;
    emaTe = emaSmoother * emaTe + (1 - emaSmoother) * teLossVal;

    trainCurve.push(emaTr);
    testCurve.push(emaTe);

    if (hasThr && timekey === null && emaTr < thr) {
      timekey = step;
    }


    // grabs
    let lastGrabResults = null;
    let grabError = null;

    if (grabs.length && step % safeGrabEvery === 0) {
      lastGrabResults = {};
      const ctx = { Xprobe, yprobe };
      for (const g0 of grabs) {
        const g = { ...g0 };
        g.k = clampInt(g.k ?? 5, 1, CAPS.kMax);
        g.layer = clampInt(g.layer ?? 0, 0, 1e9);

        const id = String(g.id ?? `${g.target}:${g.summary}:${g.layer}:${g.k}:${g.centered ? 1 : 0}`);
        try {
          const value = await computeGrab(model, grabCache, g, ctx);
          if (!grabSeries[id]) grabSeries[id] = [];
          grabSeries[id].push({ step, value });
          lastGrabResults[id] = value;
        } catch (e) {
          grabError = `Grab "${id}" failed: ${e?.message ?? String(e)}`;
        }
      }
    }

    onProgress({
      step,
      emaTrain: emaTr,
      emaTest: emaTe,
      lossThreshold: thr,
      timekey,
      lastGrabResults,
      grabError,
      paused: false,
      done: hasThr ? (timekey !== null) : (step === maxIter - 1),
    });


    if (hasThr && timekey !== null) break;
    await tf.nextFrame();
  }

  Xprobe.dispose();
  yprobe.dispose();

  return {
    timekey,
    lossThreshold: thr,
    trainCurve,
    testCurve,
    grabSeries,
    final: { emaTrain: emaTr, emaTest: emaTe },
    aborted: !!control?.isAborted?.(),
  };
}
