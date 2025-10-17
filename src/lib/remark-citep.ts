import { visit } from "unist-util-visit";
import type { Plugin } from "unified";

const remarkCitep: Plugin = () => (tree: any) => {
  visit(tree, "text", (node: any) => {
    // \citep{a,b}  ->  [@a; @b]
    node.value = node.value.replace(/\\citep\{([^}]+)\}/g, (_: any, keys: string) =>
      `[${keys.split(",").map(k => `@${k.trim()}`).join("; ")}]`
    );
    // \citet{a,b}  ->  @a; @b
    node.value = node.value.replace(/\\citet\{([^}]+)\}/g, (_: any, keys: string) =>
      keys.split(",").map(k => `@${k.trim()}`).join("; ")
    );
  });
};
export default remarkCitep;
