import assert from "node:assert/strict";
import test from "node:test";
import { parseDiff, splitRows } from "./parseDiff.ts";
import { fuzzyMatch, fuzzyFilter } from "./fuzzy.ts";

const DIFF = [
  "diff --git a/a.ts b/a.ts",
  "--- a/a.ts",
  "+++ b/a.ts",
  "@@ -1,3 +1,3 @@",
  " one",
  "-two",
  "+TWO",
  " three",
  "@@ -20,2 +20,3 @@",
  " x",
  "+y",
  " z",
].join("\n");

test("parseDiff numbers lines and counts changes", () => {
  const parsed = parseDiff(DIFF);
  assert.equal(parsed.hunks.length, 2);
  assert.equal(parsed.added, 2);
  assert.equal(parsed.deleted, 1);
  const first = parsed.hunks[0].lines;
  assert.deepEqual(first.map((l) => [l.kind, l.oldNo, l.newNo]), [
    ["ctx", 1, 1], ["del", 2, undefined], ["add", undefined, 2], ["ctx", 3, 3],
  ]);
  assert.equal(parsed.hunks[1].gapBefore, 16);
  assert.equal(parsed.hunks[1].index, 1);
});

test("splitRows pairs a replaced line", () => {
  const rows = splitRows(parseDiff(DIFF).hunks[0].lines);
  assert.equal(rows.length, 3);
  assert.equal(rows[1].left?.text, "two");
  assert.equal(rows[1].right?.text, "TWO");
});

test("fuzzyMatch prefers file-name hits and rejects non-subsequences", () => {
  assert.equal(fuzzyMatch("zzz", "src/app.ts"), null);
  const ranked = fuzzyFilter("app", ["docs/approach/readme.md", "src/app.ts"], (s) => s);
  assert.equal(ranked[0].item, "src/app.ts");
  assert.deepEqual(fuzzyMatch("ap", "app")?.indices, [0, 1]);
});
