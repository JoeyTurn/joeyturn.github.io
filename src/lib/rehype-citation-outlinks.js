import { visit } from "unist-util-visit";
import fs from "node:fs";
import path from "node:path";

// normalize (lowercase + strip accents)
const norm = (s) =>
  String(s).normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase();

// slug form used in ids (remove non a–z0–9)
const slug = (s) => norm(s).replace(/[^a-z0-9]/g, "");

// Build key -> external link map (prefer DOI over URL), storing raw/norm/slug
function buildKeyToLink(bibText) {
  const map = new Map();
  const entryRe = /@[\w]+\s*\{\s*([^,\s]+)\s*,([\s\S]*?)\}\s*(?=@|$)/gi;
  let m;
  while ((m = entryRe.exec(bibText))) {
    const rawKey = m[1].trim();
    const body = m[2];
    const doi = /doi\s*=\s*\{([^}]+)\}/i.exec(body)?.[1]?.trim();
    const url = /url\s*=\s*\{([^}]+)\}/i.exec(body)?.[1]?.trim();
    const link = doi ? `https://doi.org/${doi}` : url;
    if (!link) continue;

    map.set(rawKey, link);
    map.set(norm(rawKey), link);
    map.set(slug(rawKey), link);
  }
  return map;
}

/** Replace in-text citation anchors with external DOI/URL targets. */
export default function rehypeCitationOutlinks(opts = {}) {
  const bibPath = path.resolve(opts.bibliography || "src/content/refs.bib");
  let keyToLink = new Map();
  try {
    keyToLink = buildKeyToLink(fs.readFileSync(bibPath, "utf8"));
  } catch (e) {
    console.warn(`[citation-outlinks] Could not read ${bibPath}: ${e?.message}`);
  }

  return (tree) => {
    visit(tree, "element", (node) => {
      if (node.tagName !== "a" || !node.properties) return;

      const href = node.properties.href;
      const dataCite =
        node.properties["data-cite"] ||
        node.properties["data-citation-id"] ||
        node.properties["data-citation-item"] ||
        node.properties["data-citation-items"];

      let key = null;

      if (typeof dataCite === "string") {
        // sometimes a JSON array string: '["key"]'
        try {
          const parsed = JSON.parse(dataCite);
          key = Array.isArray(parsed) ? parsed[0] : parsed;
        } catch {
          key = dataCite;
        }
      } else if (typeof href === "string" && href.startsWith("#")) {
        const frag = decodeURIComponent(href);
        // e.g. #bib-yang:2023-spectral-scaling, #ref-Key, #bibliography-Key
        key = frag.replace(/^#*(?:bib|ref|bibliography)-/i, "");
      }

      if (!key) return;

      const out =
        keyToLink.get(key) ||
        keyToLink.get(norm(key)) ||
        keyToLink.get(slug(key));

      if (out) {
        node.properties.href = out;
        node.properties.target = "_blank";
        node.properties.rel = ["noopener", "noreferrer"];
      }
    });
  };
}
