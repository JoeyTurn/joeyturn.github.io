// trainer.js — simplified MLP trainer with two-term Hermite teacher
import * as tf from "@tensorflow/tfjs";

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

function makeSeedStream(seed) {
  let s = (seed >>> 0) || 1;
  return function nextSeed() {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s;
  };
}

/**
 * Probabilists' Hermite polynomials He_n:
 * He_0=1, He_1=x, He_{n+1}=x·He_n - n·He_{n-1}
 */
function hermite1D(x, order) {
  // x: [B] tensor, returns [B] tensor = He_order(x)
  return tf.tidy(() => {
    if (order === 0) return tf.onesLike(x);
    if (order === 1) return x.clone();
    let h0 = tf.onesLike(x);
    let h1 = x.clone();
    for (let n = 1; n < order; n++) {
      const h2 = x.mul(h1).sub(h0.mul(tf.scalar(n)));
      h0 = h1;
      h1 = h2;
    }
    return h1;
  });
}

/**
 * Two-term target:
 *   y* = coefA * He_orderA(x_{dimA} / sqrt(eigA))
 *      + coefB * He_orderB(x_{dimB} / sqrt(eigB))
 *
 * X is raw Gaussian; eigvals[i] is the variance for dimension i.
 * We scale x_i by 1/sqrt(eigval_i) before applying Hermite so that
 * He_k(z) has nice N(0,1) statistics.
 */
export function makeSyntheticDataset({
  nTrain = 4000,
  nTest = 1000,
  dIn = 4,
  eigvals,          // Float32Array or Array of length dIn, each in (0,1]
  // target: alpha * h_orderA(x_dimA) + beta * h_orderB(x_dimB)
  coefA = 1.0,
  orderA = 1,
  dimA = 0,
  coefB = 0.0,
  orderB = 2,
  dimB = 1,
  labelNoiseStd = 0.0,
  seed = 42,
} = {}) {
  const d = Math.max(1, Math.floor(dIn));
  const nextSeed = makeSeedStream(seed);

  // build eigval vector, default to all-ones
  const ev = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    const raw = eigvals ? Number(eigvals[i]) : 1.0;
    ev[i] = Math.max(1e-6, isFinite(raw) ? raw : 1.0);
  }

  function sampleX(n) {
    return tf.tidy(() => {
      // sample N(0, I)
      const raw = tf.randomNormal([n, d], 0, 1, "float32", nextSeed());
      // scale each column by sqrt(eigval)
      const scale = tf.tensor1d(ev).sqrt().reshape([1, d]);
      return raw.mul(scale); // [n, d] ~ N(0, diag(eigvals))
    });
  }

  function makeY(X) {
    return tf.tidy(() => {
      // normalize each column by its std = sqrt(eigval)
      const invScale = tf.tensor1d(ev).sqrt().reshape([1, d]);
      const Xnorm = X.div(invScale); // [n,d], each column ~ N(0,1)

      const dA = Math.min(Math.max(0, dimA), d - 1);
      const dB = Math.min(Math.max(0, dimB), d - 1);

      const xA = Xnorm.slice([0, dA], [-1, 1]).reshape([-1]); // [n]
      const xB = Xnorm.slice([0, dB], [-1, 1]).reshape([-1]); // [n]

      const hA = hermite1D(xA, Math.max(0, orderA)); // [n]
      const hB = hermite1D(xB, Math.max(0, orderB)); // [n]

      let y = hA.mul(tf.scalar(coefA)).add(hB.mul(tf.scalar(coefB))); // [n]
      y = y.reshape([-1, 1]); // [n,1]

      if (labelNoiseStd > 0) {
        y = y.add(tf.randomNormal([X.shape[0], 1], 0, labelNoiseStd, "float32", nextSeed()));
      }
      return y;
    });
  }

  const Xtr = sampleX(nTrain);
  const Xte = sampleX(nTest);
  const ytr = makeY(Xtr);
  const yte = makeY(Xte);

  return { train: { X: Xtr, y: ytr }, test: { X: Xte, y: yte }, dIn: d };
}

/** =========================================================
 *  MLP model
 *  ========================================================= */

const ACTIVATIONS = new Set(["relu", "tanh", "sigmoid", "gelu", "elu", "linear"]);

export function buildMLP({ dIn, dOut = 1, width = 256, depth = 2, activation = "relu", initScale = 1.0 }) {
  if (!ACTIVATIONS.has(activation)) throw new Error(`Unknown activation: ${activation}`);

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
      activation: undefined,
      kernelInitializer: kernelInit(width),
      biasInitializer: "zeros",
    })
  );

  return model;
}

/** =========================================================
 *  Safe grab panel (weights, activations, gradients)
 *  ========================================================= */

export const GRAB_TARGETS = ["weight", "activation", "gradient"];
export const GRAB_SUMMARIES = [
  "rms", "fro_norm", "spectral_norm", "trace_gram",
  "topk_singular_values", "topk_psd_eigenvalues",
  "grad_global_l2", "grad_per_layer_rms",
];

const CAPS = Object.freeze({
  grabEveryMin: 10, kMax: 32, probeMax: 128,
  exactSvdMaxDim: 512, randomizedIters: 1, powerIters: 12,
});

function clampInt(x, lo, hi) {
  x = Math.floor(Number(x));
  if (!Number.isFinite(x)) x = lo;
  return Math.max(lo, Math.min(hi, x));
}

function getDenseLayers(model) {
  return model.layers.filter((L) => L.getClassName && L.getClassName() === "Dense");
}

function getWeightMatrix(model, denseLayerIndex) {
  const denseLayers = getDenseLayers(model);
  if (denseLayerIndex < 0 || denseLayerIndex >= denseLayers.length) {
    throw new Error(`Layer index ${denseLayerIndex} out of range.`);
  }
  return denseLayers[denseLayerIndex].getWeights()[0];
}

function getActivationModelCached(cache, model, denseLayerIndex) {
  const key = `act_${denseLayerIndex}`;
  if (cache[key]) return cache[key];
  const denseLayers = getDenseLayers(model);
  if (denseLayerIndex < 0 || denseLayerIndex >= denseLayers.length) {
    throw new Error(`Activation layer index ${denseLayerIndex} out of range.`);
  }
  const targetLayer = denseLayers[denseLayerIndex];
  const actModel = tf.model({ inputs: model.inputs, outputs: targetLayer.output });
  cache[key] = actModel;
  return actModel;
}

async function spectralNormPowerIter(W) {
  return tf.tidy(async () => {
    const [, n] = W.shape;
    let v = tf.randomNormal([n, 1]);
    v = v.div(v.norm().add(1e-12));
    for (let i = 0; i < CAPS.powerIters; i++) {
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
  if (!tf.linalg.qr) throw new Error("Approx SVD requires tf.linalg.qr.");
  const p = 5, r = Math.min(n, k + p);
  const vals = await tf.tidy(async () => {
    let Y = W.matMul(tf.randomNormal([n, r]));
    for (let i = 0; i < CAPS.randomizedIters; i++) Y = W.matMul(W.transpose().matMul(Y));
    const { q: Q } = tf.linalg.qr(Y);
    const B = Q.transpose().matMul(W);
    const sSmall = tf.linalg.svd(B, true).s;
    const sArr = Array.from(await sSmall.data());
    sArr.sort((a, b) => b - a);
    return sArr.slice(0, k);
  });
  return vals;
}

async function computeWeightGrab(model, grab) {
  const layer = clampInt(grab.layer, 0, 1e9);
  const summary = grab.summary;
  const k = clampInt(grab.k ?? 5, 1, CAPS.kMax);
  const W = getWeightMatrix(model, layer);
  if (summary === "rms") return tf.tidy(async () => (await W.square().mean().sqrt().data())[0]);
  if (summary === "fro_norm") return tf.tidy(async () => (await W.norm("fro").data())[0]);
  if (summary === "spectral_norm") return await spectralNormPowerIter(W);
  if (summary === "trace_gram") return tf.tidy(async () => (await W.square().sum().data())[0]);
  if (summary === "topk_singular_values") return await topKSingularValues(W, k);
  if (summary === "topk_psd_eigenvalues") { const s = await topKSingularValues(W, k); return s.map((x) => x * x); }
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
    if (centered) H = H.sub(H.mean(0, true));
    if (summary === "rms") return (await H.square().mean().sqrt().data())[0];
    if (summary === "trace_gram") return (await H.square().sum().data())[0];
    if (summary === "fro_norm") return (await H.norm("fro").data())[0];
    if (summary === "spectral_norm") { const s = tf.linalg.svd(H, true).s; return (await s.data())[0]; }
    if (summary === "topk_singular_values") { const s = tf.linalg.svd(H, true).s; const sArr = Array.from(await s.data()); sArr.sort((a, b) => b - a); return sArr.slice(0, k); }
    if (summary === "topk_psd_eigenvalues") { const s = tf.linalg.svd(H, true).s; const sArr = Array.from(await s.data()); sArr.sort((a, b) => b - a); return sArr.slice(0, k).map((x) => x * x); }
    throw new Error(`Unsupported activation summary: ${summary}`);
  });
}

async function computeGradientGrab(model, grab, Xprobe, yprobe) {
  const summary = grab.summary;
  const vars = model.trainableWeights.map((w) => w.val);
  return tf.tidy(async () => {
    const { grads } = tf.variableGrads(() => {
      const logits = model.apply(Xprobe, { training: false });
      const y2 = yprobe.reshape([-1, 1]);
      return tf.losses.meanSquaredError(y2, logits).mean();
    }, vars);
    const gradTensors = vars.map((v) => grads[v.name]).filter(Boolean);
    if (summary === "rms") {
      const totalSq = tf.addN(gradTensors.map((g) => g.square().sum()));
      const totalN = gradTensors.reduce((acc, g) => acc + g.size, 0);
      return (await totalSq.div(tf.scalar(Math.max(1, totalN))).sqrt().data())[0];
    }
    if (summary === "grad_global_l2") {
      return (await tf.addN(gradTensors.map((g) => g.square().sum())).sqrt().data())[0];
    }
    if (summary === "grad_per_layer_rms") {
      const denseLayers = getDenseLayers(model);
      const arr = [];
      for (let i = 0; i < denseLayers.length; i++) {
        const w = denseLayers[i].getWeights()[0];
        const g = grads[w.name];
        arr.push(g ? (await g.square().mean().sqrt().data())[0] : NaN);
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

function mseLoss(logits, y) {
  const y2 = y.reshape([-1, 1]);
  return tf.losses.meanSquaredError(y2, logits).mean();
}

function gammaRescaledLR(lr, gamma) {
  return gamma >= 1 ? lr * gamma : lr * (gamma * gamma);
}

/** =========================================================
 *  Main training loop
 *  ========================================================= */

export async function trainMLP(config, onProgress = () => {}) {
  const { data, model: modelCfg, train: trainCfg } = config;

  const {
    maxIter = 100000,
    batchSize = 256,
    lr = 1e-2,
    gamma = 1.0,
    optimizer = "adam",
    emaSmoother = 0.9,
    lossThreshold = 0.05,
    grabs = [],
    grabEvery = 50,
    probeSize = 64,
    control = null,
  } = trainCfg;

  const safeGrabEvery = clampInt(grabEvery, CAPS.grabEveryMin, 1e9);
  const safeProbe = clampInt(probeSize, 8, CAPS.probeMax);

  const ds = makeSyntheticDataset({
    nTrain: data.nTrain ?? 4000,
    nTest: data.nTest ?? 1000,
    dIn: data.dIn ?? 4,
    eigvals: data.eigvals,
    coefA: data.coefA ?? 1.0,
    orderA: data.orderA ?? 1,
    dimA: data.dimA ?? 0,
    coefB: data.coefB ?? 0.0,
    orderB: data.orderB ?? 2,
    dimB: data.dimB ?? 1,
    labelNoiseStd: data.labelNoiseStd ?? 0.0,
    seed: data.seed ?? 42,
  });

  const Xtr = ds.train.X;
  const ytr = ds.train.y;
  const Xte = ds.test.X;
  const yte = ds.test.y;
  const dIn = ds.dIn;

  const model = buildMLP({
    dIn,
    dOut: 1,
    width: modelCfg.width ?? 256,
    depth: modelCfg.depth ?? 2,
    activation: modelCfg.activation ?? "relu",
    initScale: modelCfg.initScale ?? 1.0,
  });

  const effLR = gammaRescaledLR(lr, gamma);
  const opt = optimizer === "adam" ? tf.train.adam(effLR) : tf.train.sgd(effLR);

  const thr = Number(lossThreshold);
  const hasThr = Number.isFinite(thr) && thr > 0;
  let timekey = null;

  const nTrain = Xtr.shape[0];
  const probeN = Math.min(safeProbe, nTrain);
  const Xprobe = Xtr.slice([0, 0], [probeN, dIn]);
  const yprobe = ytr.slice([0, 0], [probeN, 1]);
  const grabCache = {};

  let emaTr = null, emaTe = null;

  function batchIndices(step) {
    const start = (step * batchSize) % nTrain;
    const end = Math.min(nTrain, start + batchSize);
    return { start, end };
  }

  for (let step = 0; step < maxIter; step++) {
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

    if (step === 0) { emaTr = trLossVal; emaTe = teLossVal; }
    emaTr = emaSmoother * emaTr + (1 - emaSmoother) * trLossVal;
    emaTe = emaSmoother * emaTe + (1 - emaSmoother) * teLossVal;

    if (hasThr && timekey === null && emaTr < thr) timekey = step;

    let lastGrabResults = null, grabError = null;
    if (grabs.length && step % safeGrabEvery === 0) {
      lastGrabResults = {};
      const ctx = { Xprobe, yprobe };
      for (const g0 of grabs) {
        const g = { ...g0 };
        g.k = clampInt(g.k ?? 5, 1, CAPS.kMax);
        g.layer = clampInt(g.layer ?? 0, 0, 1e9);
        const id = String(g.id ?? `${g.target}:${g.summary}:${g.layer}:${g.k}`);
        try {
          lastGrabResults[id] = await computeGrab(model, grabCache, g, ctx);
        } catch (e) {
          grabError = `Grab "${id}" failed: ${e?.message ?? String(e)}`;
        }
      }
    }

    onProgress({
      step,
      rawTrain: trLossVal,
      rawTest: teLossVal,
      emaTrain: emaTr,
      emaTest: emaTe,
      lossThreshold: thr,
      timekey,
      lastGrabResults,
      grabError,
      paused: false,
      done: hasThr ? timekey !== null : step === maxIter - 1,
    });

    if (hasThr && timekey !== null) break;
    await tf.nextFrame();
  }

  Xprobe.dispose();
  yprobe.dispose();

  return {
    timekey, lossThreshold: thr,
    final: { emaTrain: emaTr, emaTest: emaTe },
    aborted: !!control?.isAborted?.(),
  };
}
