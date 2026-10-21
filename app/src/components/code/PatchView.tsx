import { useEffect, useState } from "react";

interface PatchViewProps {
  diff: string;
  truncated?: boolean;
  onUndoHunk?: (hunk: number) => void;
  onAcceptHunk?: (hunk: number) => void;
}

/** Coloured unified diff. Hunk indexes match the order `git apply` sees. */
export function PatchView({ diff, truncated, onUndoHunk, onAcceptHunk }: PatchViewProps) {
  const [kept, setKept] = useState<Record<number, boolean>>({});
  useEffect(() => { setKept({}); }, [diff]);
  const lines = diff.split("\n");
  let hunk = -1;
  return (
    <div className="font-mono text-[11px] bg-th-inset-bg rounded-lg p-2 overflow-x-auto">
      {lines.map((line, index) => {
        const isHunk = line.startsWith("@@");
        if (isHunk) hunk += 1;
        const hunkIndex = hunk;
        const added = line.startsWith("+") && !line.startsWith("+++");
        const removed = line.startsWith("-") && !line.startsWith("---");
        const color = added
          ? "text-emerald-400"
          : removed
            ? "text-red-300"
            : isHunk
              ? "text-sky-300"
              : "text-th-text-muted";
        return (
          <div key={index} className="flex items-start gap-2 min-w-0">
            {isHunk && !truncated && (onUndoHunk || onAcceptHunk) ? (
              <span className="shrink-0 flex items-center gap-1">
                {onAcceptHunk && (
                  <button
                    type="button"
                    className="text-[10px] text-th-text-muted hover:text-emerald-300"
                    onClick={() => {
                      setKept((prev) => ({ ...prev, [hunkIndex]: true }));
                      onAcceptHunk(hunkIndex);
                    }}
                  >
                    {kept[hunkIndex] ? "Kept" : "Accept"}
                  </button>
                )}
                {onUndoHunk && !kept[hunkIndex] && (
                  <button
                    type="button"
                    className="text-[10px] text-th-text-muted hover:text-red-300"
                    onClick={() => onUndoHunk(hunkIndex)}
                  >
                    Reject
                  </button>
                )}
              </span>
            ) : null}
            <span className={`whitespace-pre-wrap break-all ${color}`}>{line || " "}</span>
          </div>
        );
      })}
      {truncated && (
        <p className="mt-2 text-th-text-muted">Diff truncated. The rest is on disk.</p>
      )}
    </div>
  );
}
