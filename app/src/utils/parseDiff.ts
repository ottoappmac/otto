export type DiffLineKind = "add" | "del" | "ctx";

export interface DiffLine {
  kind: DiffLineKind;
  text: string;
  oldNo?: number;
  newNo?: number;
}

export interface DiffHunk {
  /** Matches the order `git apply` sees. */
  index: number;
  header: string;
  lines: DiffLine[];
  /** Unchanged lines between the previous hunk and this one. */
  gapBefore: number;
}

export interface ParsedDiff {
  hunks: DiffHunk[];
  added: number;
  deleted: number;
}

const HUNK = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/;

/** Parse a single-file unified diff. File headers and "\ No newline" notes are dropped. */
export function parseDiff(diff: string): ParsedDiff {
  const hunks: DiffHunk[] = [];
  let current: DiffHunk | null = null;
  let oldNo = 0;
  let newNo = 0;
  let added = 0;
  let deleted = 0;
  let prevEnd = 0;
  for (const line of diff.split("\n")) {
    const match = HUNK.exec(line);
    if (match) {
      oldNo = Number(match[1]);
      newNo = Number(match[2]);
      current = { index: hunks.length, header: line, lines: [], gapBefore: Math.max(0, oldNo - prevEnd - 1) };
      hunks.push(current);
      continue;
    }
    if (!current) continue;
    if (line.startsWith("\\")) continue;
    if (line.startsWith("+")) {
      current.lines.push({ kind: "add", text: line.slice(1), newNo });
      newNo += 1;
      added += 1;
    } else if (line.startsWith("-")) {
      current.lines.push({ kind: "del", text: line.slice(1), oldNo });
      oldNo += 1;
      deleted += 1;
    } else if (line.startsWith(" ")) {
      current.lines.push({ kind: "ctx", text: line.slice(1), oldNo, newNo });
      oldNo += 1;
      newNo += 1;
    } else {
      continue;
    }
    prevEnd = oldNo - 1;
  }
  return { hunks, added, deleted };
}

export interface SplitRow {
  left?: DiffLine;
  right?: DiffLine;
}

/** Pair removed and added runs side by side for the split view. */
export function splitRows(lines: DiffLine[]): SplitRow[] {
  const rows: SplitRow[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.kind === "ctx") {
      rows.push({ left: line, right: line });
      i += 1;
      continue;
    }
    const dels: DiffLine[] = [];
    const adds: DiffLine[] = [];
    while (i < lines.length && lines[i].kind === "del") dels.push(lines[i++]);
    while (i < lines.length && lines[i].kind === "add") adds.push(lines[i++]);
    const n = Math.max(dels.length, adds.length);
    for (let k = 0; k < n; k += 1) rows.push({ left: dels[k], right: adds[k] });
  }
  return rows;
}
