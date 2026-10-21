import type { FolderGrantPreview } from "../../types";

export function FolderGrantDialog({
  preview,
  mode,
  busy,
  onMode,
  onCancel,
  onConfirm,
}: {
  preview: FolderGrantPreview;
  mode: "read" | "write";
  busy?: boolean;
  onMode: (mode: "read" | "write") => void;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div className="w-full max-w-lg rounded-2xl border border-th-border bg-th-bg-secondary p-5 shadow-xl">
        <h2 className="text-sm font-semibold text-th-text-primary">Allow this folder?</h2>
        <p className="mt-2 text-xs text-th-text-secondary break-all font-mono">{preview.realpath}</p>
        <p className="mt-3 text-xs leading-relaxed text-th-text-tertiary">{preview.coverage}</p>
        <div className="mt-4 flex gap-2">
          <button
            type="button"
            onClick={() => onMode("read")}
            className={`px-3 py-1.5 rounded-lg text-xs border ${mode === "read" ? "border-sky-400/50 text-sky-300 bg-sky-500/10" : "border-th-border text-th-text-secondary"}`}
          >
            Read only
          </button>
          <button
            type="button"
            onClick={() => onMode("write")}
            className={`px-3 py-1.5 rounded-lg text-xs border ${mode === "write" ? "border-sky-400/50 text-sky-300 bg-sky-500/10" : "border-th-border text-th-text-secondary"}`}
          >
            Edit, with checkpoints
          </button>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <button type="button" onClick={onCancel} className="px-3 py-1.5 text-xs text-th-text-secondary">
            Cancel
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={onConfirm}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-sky-500/20 text-sky-200 border border-sky-500/30 disabled:opacity-50"
          >
            {busy ? "Allowing…" : "Allow folder"}
          </button>
        </div>
      </div>
    </div>
  );
}
