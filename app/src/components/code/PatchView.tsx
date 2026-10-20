interface PatchViewProps {
  diff: string;
  truncated?: boolean;
  onUndoHunk?: (hunk: number) => void;
}

/** Coloured unified diff. Hunk indexes match the order `git apply` sees. */
export function PatchView({ diff, truncated, onUndoHunk }: PatchViewProps) {
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
            {isHunk && onUndoHunk && !truncated ? (
              <button
                type="button"
                className="shrink-0 text-[10px] text-th-text-muted hover:text-th-text-primary"
                onClick={() => onUndoHunk(hunkIndex)}
              >
                Undo
              </button>
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
