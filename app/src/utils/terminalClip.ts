/** Terminal text that travels to the chat as a chip instead of loose pasted text. */

export interface Snippet {
  id: string;
  /** Where it came from, e.g. "Shell" or "npm run dev". */
  label: string;
  text: string;
  lines: number;
}

const MAX_CHARS = 20_000;
let lastCopy = "";
let counter = 0;

function normalize(text: string): string {
  return text.replace(/[ \t]+$/gm, "").trim();
}

/** Remember what the user just copied from a terminal. */
export function rememberTerminalCopy(text: string): void {
  lastCopy = text;
}

/** A short one-line paste stays inline. Several lines or a long run becomes a chip. */
export function qualifiesAsSnippet(text: string): boolean {
  const body = normalize(text);
  return body.includes("\n") || body.length > 80;
}

/** True when *pasted* is the text last copied out of a terminal, and big enough for a chip. */
export function isTerminalPaste(pasted: string): boolean {
  return lastCopy !== "" && normalize(pasted) === normalize(lastCopy) && qualifiesAsSnippet(pasted);
}

export function makeSnippet(label: string, text: string): Snippet {
  let body = normalize(text);
  if (body.length > MAX_CHARS) body = `… (earlier output cut)\n${body.slice(-MAX_CHARS)}`;
  counter += 1;
  return { id: `snip-${Date.now()}-${counter}`, label, text: body, lines: body.split("\n").length };
}

/** The message prefix that carries the snippets to the agent. */
export function formatSnippets(snippets: Snippet[]): string {
  return snippets
    .map((snippet) => {
      const fence = snippet.text.includes("```") ? "~~~~" : "```";
      return `[Terminal output: ${snippet.label}]\n${fence}\n${snippet.text}\n${fence}`;
    })
    .join("\n\n");
}
