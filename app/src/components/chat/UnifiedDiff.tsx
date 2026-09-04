import { useMemo } from "react";

interface UnifiedDiffProps {
  path: string;
  oldText: string;
  newText: string;
  onOpen?: () => void;
  /** Cap rendered lines (old + new). */
  maxLines?: number;
}

/** Search-replace visualization: removed old_string, added new_string. */
export function UnifiedDiff({ path, oldText, newText, onOpen, maxLines = 80 }: UnifiedDiffProps) {
  const { oldLines, newLines, omitted } = useMemo(() => {
    const o = oldText.split("\n");
    const n = newText.split("\n");
    const budget = Math.max(4, maxLines);
    const oldKeep = Math.min(o.length, Math.ceil(budget / 2));
    const newKeep = Math.min(n.length, budget - oldKeep);
    return {
      oldLines: o.slice(0, oldKeep),
      newLines: n.slice(0, newKeep),
      omitted: Math.max(0, o.length - oldKeep) + Math.max(0, n.length - newKeep),
    };
  }, [oldText, newText, maxLines]);

  const filename = path.split("/").pop() ?? path;

  return (
    <div className="mt-1.5 ml-6 rounded-lg border border-th-border bg-th-card-bg overflow-hidden max-w-2xl">
      <button
        type="button"
        onClick={onOpen}
        className="w-full flex items-center gap-2 px-3 py-1.5 bg-th-inset-bg border-b border-th-border text-left hover:bg-th-surface-hover transition-colors"
        title={path}
      >
        <span className="text-[11px] font-mono text-th-text-primary truncate">{filename}</span>
        <span className="text-[10px] text-th-text-muted truncate flex-1">{path}</span>
        {onOpen && (
          <span className="text-[10px] text-blue-400 shrink-0">Open</span>
        )}
      </button>
      <pre className="text-[11px] leading-[1.55] font-mono overflow-x-auto max-h-64 overflow-y-auto">
        {oldLines.map((line, i) => (
          <div key={`-${i}`} className="px-3 bg-red-500/10 text-red-300/90 whitespace-pre">
            <span className="select-none text-red-400/70 mr-2">-</span>
            {line || " "}
          </div>
        ))}
        {newLines.map((line, i) => (
          <div key={`+${i}`} className="px-3 bg-emerald-500/10 text-emerald-300/90 whitespace-pre">
            <span className="select-none text-emerald-400/70 mr-2">+</span>
            {line || " "}
          </div>
        ))}
        {omitted > 0 && (
          <div className="px-3 py-1 text-[10px] text-th-text-muted bg-th-inset-bg">
            … {omitted} more lines
          </div>
        )}
      </pre>
    </div>
  );
}
