/** Syntax-highlight a code string for a file path. Falls back to escaped text. */

import hljs from "highlight.js/lib/common";

const EXT_LANG: Record<string, string> = {
  py: "python", pyw: "python",
  js: "javascript", mjs: "javascript", cjs: "javascript",
  jsx: "javascript",
  ts: "typescript", tsx: "typescript",
  sh: "bash", bash: "bash", zsh: "bash",
  rb: "ruby", go: "go", rs: "rust",
  java: "java", kt: "kotlin", swift: "swift",
  c: "c", h: "c", cpp: "cpp", cc: "cpp", hpp: "cpp",
  cs: "csharp", php: "php", lua: "lua", r: "r",
  css: "css", scss: "scss", less: "less",
  json: "json", yaml: "yaml", yml: "yaml",
  toml: "toml", ini: "ini", sql: "sql",
  md: "markdown", html: "xml", xml: "xml",
  dockerfile: "dockerfile", makefile: "makefile",
  graphql: "graphql", proto: "protobuf",
};

export function languageFromPath(path: string): string | undefined {
  const base = path.split("/").pop()?.toLowerCase() ?? "";
  if (base === "dockerfile" || base === "makefile") return EXT_LANG[base];
  const ext = base.includes(".") ? base.split(".").pop() ?? "" : "";
  return EXT_LANG[ext];
}

export function highlightCode(code: string, path?: string): string {
  try {
    const lang = path ? languageFromPath(path) : undefined;
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
    }
    return hljs.highlightAuto(code).value;
  } catch {
    return escapeHtml(code);
  }
}

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
