// src/mlp-trainer/boot.js

async function boot() {
  const nodes = document.querySelectorAll("[data-mlp-trainer]");
  if (!nodes.length) return;

  // Resolve relative to this module (works after bundling on GH Pages)
  const { mountMLPTrainer } = await import(new URL("./mount.js", import.meta.url).href);

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
  document.addEventListener("DOMContentLoaded", () => boot());
} else {
  boot();
}
