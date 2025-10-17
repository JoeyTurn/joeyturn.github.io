import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeCitation from "rehype-citation";
import remarkCitep from "./src/lib/remark-citep.js";
import remarkFootnotes from "remark-footnotes";

export default defineConfig({
  site: "http://localhost:4321",
  markdown: {
    remarkPlugins: [
      remarkMath,
      remarkCitep,
    ],
    rehypePlugins: [
      [remarkFootnotes, { inlineNotes: true }],
      [rehypeKatex, { trust: false }],
      [rehypeCitation, {
        bibliography: "src/content/refs.bib",  // or per-page (see below)
        linkCitations: true,                   // link in-text → bibliography
      }],
    ],
  },
});