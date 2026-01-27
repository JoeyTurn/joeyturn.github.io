import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
// import rehypeKatex from "rehype-katex";
import rehypeCitation from "rehype-citation";
import rehypeCitationOutlinks from "./src/lib/rehype-citation-outlinks.js";
import remarkCitep from "./src/lib/remark-citep.js";
import remarkFootnotes from "remark-footnotes";
import katexPerPage from "./src/lib/rehype-katex-per-page.js";
import mdx from "@astrojs/mdx";

export default defineConfig({
  site: "http://localhost:4321",
  markdown: {
    remarkPlugins: [
      remarkMath,
      remarkCitep,
      [remarkFootnotes, { inlineNotes: true }],
    ],
    rehypePlugins: [
    //   [rehypeKatex, { trust: false }],
      [rehypeCitation, {
        bibliography: "src/content/refs.bib",  // or per-page (see below)
        linkCitations: true,                   // link in-text → bibliography
      }],
      [katexPerPage, {
        macros: {
             "\\RR": "\\mathbb{R}"
            }
      }],
      [rehypeCitationOutlinks, { bibliography: "src/content/refs.bib" }],
    ],
  },
  integrations: [mdx()],
  base: "/blog/mlp_train_widget",
  vite: {
    build: {
      assetsInlineLimit: 0, // <-- prevents data: URLs for small JS assets
    },
  },
});