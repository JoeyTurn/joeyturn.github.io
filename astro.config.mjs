import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeCitation from "rehype-citation";

export default defineConfig({
  site: "http://localhost:4321",
  markdown: {
    remarkPlugins: [
      remarkMath,                // $...$ and $$...$$
    ],
    rehypePlugins: [
      [rehypeKatex, { trust: false }],
      [rehypeCitation, {
        bibliography: "src/content/refs.bib",  // or per-page (see below)
        linkCitations: true,                   // link in-text → bibliography
      }],
    ],
  },
});