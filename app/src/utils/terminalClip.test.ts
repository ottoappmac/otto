import assert from "node:assert/strict";
import test from "node:test";
import { formatSnippets, isTerminalPaste, makeSnippet, qualifiesAsSnippet, rememberTerminalCopy } from "./terminalClip.ts";

test("only multi-line or long text becomes a chip", () => {
  assert.equal(qualifiesAsSnippet("npm test"), false);
  assert.equal(qualifiesAsSnippet("line one\nline two"), true);
  assert.equal(qualifiesAsSnippet("x".repeat(81)), true);
});

test("a paste matches the last terminal copy despite trailing spaces", () => {
  rememberTerminalCopy("Error: boom   \n  at app.js:3  \n");
  assert.equal(isTerminalPaste("Error: boom\n  at app.js:3"), true);
  assert.equal(isTerminalPaste("something else\nentirely"), false);
});

test("a short terminal copy stays inline", () => {
  rememberTerminalCopy("ls -la");
  assert.equal(isTerminalPaste("ls -la"), false);
});

test("snippets are fenced and long text is cut from the front", () => {
  const snippet = makeSnippet("Shell", "a\nb");
  assert.equal(snippet.lines, 2);
  assert.equal(formatSnippets([snippet]), "[Terminal output: Shell]\n```\na\nb\n```");
  const big = makeSnippet("Shell", "z".repeat(30_000));
  assert.ok(big.text.startsWith("… (earlier output cut)"));
  assert.ok(big.text.length < 20_100);
  const tricky = makeSnippet("Shell", "```js\nx\n```");
  assert.ok(formatSnippets([tricky]).includes("~~~~"));
});
