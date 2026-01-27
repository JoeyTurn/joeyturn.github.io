// src/mlp-trainer/boot.js
import { mountMLPTrainer } from "./mount.js";

function boot() {
  const nodes = document.querySelectorAll("[data-mlp-trainer]");
  for (const node of nodes) {
    if (node.__mlpMounted) continue;
    node.__mlpMounted = true;
    try {
      mountMLPTrainer(node);
    } catch (e) {
      console.error("[MLPTrainer] mount failed:", e);
      node.innerHTML = `<pre style="color:#b00; white-space:pre-wrap;">${String(e?.stack || e)}</pre>`;
    }
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
