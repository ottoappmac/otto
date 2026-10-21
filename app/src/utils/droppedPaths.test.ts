import assert from "node:assert/strict";
import test from "node:test";
import { splitDroppedPaths } from "./droppedPaths.ts";

test("a dotless name is a file unless stat says it is a directory", () => {
  const split = splitDroppedPaths(
    ["/proj/Makefile", "/proj/my.app", "/proj/src"],
    [
      { path: "/proj/Makefile", is_dir: false, is_file: true },
      { path: "/proj/my.app", is_dir: true, is_file: false },
      { path: "/proj/src", is_dir: true, is_file: false },
    ],
  );
  assert.deepEqual(split.files, ["/proj/Makefile"]);
  assert.deepEqual(split.folders, ["/proj/my.app", "/proj/src"]);
});

test("without a stat result, only a trailing slash is a folder", () => {
  const split = splitDroppedPaths(["/proj/Makefile", "/proj/src/"], null);
  assert.deepEqual(split.files, ["/proj/Makefile"]);
  assert.deepEqual(split.folders, ["/proj/src/"]);
});
