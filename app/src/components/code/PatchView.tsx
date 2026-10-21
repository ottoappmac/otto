import { memo, useEffect, useMemo, useState } from "react";
import { Check, ChevronsUpDown, Columns2, Rows2, Undo2 } from "lucide-react";
import { highlightCode } from "../../utils/highlightCode";
import { parseDiff, splitRows, type DiffLine } from "../../utils/parseDiff";
import { IconButton } from "../ui/Button";

interface PatchViewProps {
  diff: string;
  /** File path, used to pick the syntax highlighter. */
  path?: string;
  truncated?: boolean;
  onUndoHunk?: (hunk: number) => void;
  onAcceptHunk?: (hunk: number) => void;
}

type Layout = "unified" | "split";
const LAYOUT_KEY = "code.diffLayout";

function storedLayout(): Layout {
  try {
    return localStorage.getItem(LAYOUT_KEY) === "split" ? "split" : "unified";
  } catch {
    return "unified";
  }
}

const ROW_BG = {
  add: "bg-th-diff-add/12",
  del: "bg-th-diff-del/12",
  ctx: "",
} as const;
const GUTTER_BG = {
  add: "bg-th-diff-add/20 text-th-diff-add",
  del: "bg-th-diff-del/20 text-th-diff-del",
  ctx: "text-th-text-faint",
} as const;
const SIGN = { add: "+", del: "−", ctx: " " } as const;

const Code = memo(function Code({ text, path }: { text: string; path?: string }) {
  const html = useMemo(() => (text ? highlightCode(text, path) : " "), [text, path]);
  return <span className="hljs whitespace-pre-wrap break-all" dangerouslySetInnerHTML={{ __html: html }} />;
});

function UnifiedRow({ line, path }: { line: DiffLine; path?: string }) {
  return (
    <div className={`flex ${ROW_BG[line.kind]}`}>
      <span className={`w-9 shrink-0 select-none pr-1.5 text-right ${GUTTER_BG[line.kind]}`}>{line.oldNo ?? ""}</span>
      <span className={`w-9 shrink-0 select-none pr-1.5 text-right ${GUTTER_BG[line.kind]}`}>{line.newNo ?? ""}</span>
      <span className={`w-4 shrink-0 select-none text-center ${GUTTER_BG[line.kind]}`}>{SIGN[line.kind]}</span>
      <div className="min-w-0 flex-1 pl-1.5 pr-2"><Code text={line.text} path={path} /></div>
    </div>
  );
}

function SplitCell({ line, side, path }: { line?: DiffLine; side: "left" | "right"; path?: string }) {
  if (!line) return <div className="flex-1 basis-0 bg-th-bg-tertiary/40" />;
  const no = side === "left" ? line.oldNo : line.newNo;
  return (
    <div className={`flex min-w-0 flex-1 basis-0 ${ROW_BG[line.kind]}`}>
      <span className={`w-9 shrink-0 select-none pr-1.5 text-right ${GUTTER_BG[line.kind]}`}>{no ?? ""}</span>
      <div className="min-w-0 flex-1 pl-1.5 pr-2"><Code text={line.text} path={path} /></div>
    </div>
  );
}

/** Syntax-highlighted unified or split diff. Hunk indexes match the order `git apply` sees. */
export function PatchView({ diff, path, truncated, onUndoHunk, onAcceptHunk }: PatchViewProps) {
  const [kept, setKept] = useState<Record<number, boolean>>({});
  const [layout, setLayout] = useState<Layout>(storedLayout);
  useEffect(() => { setKept({}); }, [diff]);
  const parsed = useMemo(() => parseDiff(diff), [diff]);

  const chooseLayout = (next: Layout) => {
    setLayout(next);
    try { localStorage.setItem(LAYOUT_KEY, next); } catch { /* storage unavailable */ }
  };

  if (parsed.hunks.length === 0) {
    return <p className="text-th-text-muted">No line changes to show.</p>;
  }

  const actionable = !truncated && (onUndoHunk || onAcceptHunk);

  return (
    <div className="overflow-hidden rounded-lg border border-th-border bg-th-inset-bg">
      <div className="flex items-center gap-2 border-b border-th-border px-2 py-1 text-2xs">
        <span className="text-th-diff-add">+{parsed.added}</span>
        <span className="text-th-diff-del">−{parsed.deleted}</span>
        <span className="text-th-text-muted">{parsed.hunks.length} change{parsed.hunks.length === 1 ? "" : "s"}</span>
        <span className="ml-auto flex items-center gap-0.5">
          <IconButton
            label="Unified view"
            aria-pressed={layout === "unified"}
            icon={<Rows2 size={13} />}
            onClick={() => chooseLayout("unified")}
            className={layout === "unified" ? "!bg-th-surface-hover !text-th-text-primary" : ""}
          />
          <IconButton
            label="Split view"
            aria-pressed={layout === "split"}
            icon={<Columns2 size={13} />}
            onClick={() => chooseLayout("split")}
            className={layout === "split" ? "!bg-th-surface-hover !text-th-text-primary" : ""}
          />
        </span>
      </div>
      <div className="overflow-x-auto font-mono text-2xs leading-[1.55]">
        {parsed.hunks.map((hunk) => (
          <section key={hunk.index} aria-label={`Change ${hunk.index + 1}`}>
            {hunk.gapBefore > 0 && (
              <div className="flex items-center gap-1.5 bg-th-bg-tertiary/60 px-2 py-0.5 text-th-text-muted">
                <ChevronsUpDown size={11} /> {hunk.gapBefore} unchanged line{hunk.gapBefore === 1 ? "" : "s"}
              </div>
            )}
            <div className="sticky left-0 flex items-center gap-2 bg-th-accent/8 px-2 py-0.5 text-th-accent">
              <span className="min-w-0 flex-1 truncate">{hunk.header}</span>
              {actionable && (
                <span className="flex shrink-0 items-center gap-0.5 font-sans">
                  {kept[hunk.index] ? (
                    <span className="flex items-center gap-1 px-1 text-th-success"><Check size={11} /> Kept</span>
                  ) : (
                    <>
                      {onAcceptHunk && (
                        <IconButton
                          label="Keep this change"
                          icon={<Check size={13} />}
                          className="hover:!text-th-success"
                          onClick={() => {
                            setKept((prev) => ({ ...prev, [hunk.index]: true }));
                            onAcceptHunk(hunk.index);
                          }}
                        />
                      )}
                      {onUndoHunk && (
                        <IconButton
                          label="Reject this change"
                          icon={<Undo2 size={13} />}
                          className="hover:!text-th-danger"
                          onClick={() => onUndoHunk(hunk.index)}
                        />
                      )}
                    </>
                  )}
                </span>
              )}
            </div>
            {layout === "unified"
              ? hunk.lines.map((line, i) => <UnifiedRow key={i} line={line} path={path} />)
              : splitRows(hunk.lines).map((row, i) => (
                  <div key={i} className="flex divide-x divide-th-border">
                    <SplitCell line={row.left} side="left" path={path} />
                    <SplitCell line={row.right} side="right" path={path} />
                  </div>
                ))}
          </section>
        ))}
      </div>
      {truncated && <p className="border-t border-th-border px-2 py-1 text-2xs text-th-text-muted">Diff truncated. The rest is on disk.</p>}
    </div>
  );
}
