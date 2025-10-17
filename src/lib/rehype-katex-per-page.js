import rehypeKatex from "rehype-katex";

/** Merge global KaTeX options with per-file macros from frontmatter.katexMacros */
export default function katexPerPage(globalOpts = {}) {
  return (tree, file) => {
    const fm = file.data?.astro?.frontmatter ?? {};
    const fmMacros = fm.katexMacros || {};
    const opts = {
      ...globalOpts,
      macros: { ...(globalOpts.macros || {}), ...fmMacros },
    };
    return rehypeKatex(opts)(tree, file);
  };
}
