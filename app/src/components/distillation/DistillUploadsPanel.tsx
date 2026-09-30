import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  FileUp,
  Loader2,
  Trash2,
  Upload,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import type { DistillUpload, DistillValidateReport } from "../../types";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

function formatWhen(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function FormatSpec() {
  return (
    <div className="rounded-xl border border-th-border bg-th-inset-bg/40 px-4 py-3 space-y-2">
      <p className="text-[11px] font-medium text-th-text-secondary">Expected format</p>
      <p className="text-[11px] text-th-text-tertiary leading-relaxed">
        One JSON object per line (JSONL), or a JSON array. Each example needs a{" "}
        <code className="font-mono text-[10px]">messages</code> array with at least one{" "}
        <code className="font-mono text-[10px]">user</code> and one{" "}
        <code className="font-mono text-[10px]">assistant</code> turn. ShareGPT{" "}
        <code className="font-mono text-[10px]">conversations</code> is also accepted.
      </p>
      <pre className="text-[10px] font-mono text-th-text-muted bg-black/30 rounded-lg p-2.5 overflow-x-auto leading-relaxed">{`{"messages":[{"role":"user","content":"Summarize this"},{"role":"assistant","content":"…"}]}`}</pre>
    </div>
  );
}

function ReportCard({ report }: { report: DistillValidateReport }) {
  return (
    <div className={`rounded-xl border px-4 py-3 space-y-2 ${
      report.ok
        ? "border-emerald-500/25 bg-emerald-500/5"
        : "border-red-500/25 bg-red-500/5"
    }`}>
      <div className="flex items-center gap-2">
        {report.ok
          ? <CheckCircle2 size={14} className="text-emerald-400" />
          : <AlertTriangle size={14} className="text-red-400" />}
        <p className={`text-xs font-medium ${report.ok ? "text-emerald-300" : "text-red-300"}`}>
          {report.ok
            ? `${report.n_valid.toLocaleString()} valid row${report.n_valid === 1 ? "" : "s"}`
            : "Not usable for training"}
        </p>
        <span className="text-[10px] text-th-text-muted ml-auto">
          {report.container}
          {report.format ? ` · ${report.format}` : ""}
        </span>
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-th-text-tertiary">
        <span>{report.n_rows} rows</span>
        <span>{report.n_invalid} invalid</span>
        <span>{report.n_messages.toLocaleString()} messages</span>
        <span>{report.n_tool_calls} tool calls</span>
      </div>
      {report.sample_preview && (
        <p className="text-[11px] text-th-text-secondary italic truncate">
          “{report.sample_preview}”
        </p>
      )}
      {report.errors.length > 0 && (
        <ul className="space-y-1 max-h-36 overflow-y-auto">
          {report.errors.map((e, i) => (
            <li key={`${e.code}-${e.line}-${i}`} className="text-[11px] text-red-300">
              {e.line > 0 ? `Line ${e.line}: ` : ""}{e.message}
            </li>
          ))}
        </ul>
      )}
      {report.warnings.length > 0 && (
        <ul className="space-y-1 max-h-24 overflow-y-auto">
          {report.warnings.map((e, i) => (
            <li key={`${e.code}-${e.line}-${i}`} className="text-[11px] text-amber-300">
              {e.line > 0 ? `Line ${e.line}: ` : ""}{e.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function DistillUploadsPanel({
  selectedId,
  onSelect,
}: {
  selectedId: string | null;
  onSelect: (upload: DistillUpload | null) => void;
}) {
  const [uploads, setUploads] = useState<DistillUpload[]>([]);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const [report, setReport] = useState<DistillValidateReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.listDistillUploads();
      setUploads(res.uploads ?? []);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to list uploads");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const handleFiles = async (files: FileList | File[] | null) => {
    const file = files?.[0];
    if (!file) return;
    setError(null);
    setBusy(true);
    try {
      const checked = await api.validateDistillFile(file);
      setReport(checked);
      if (!checked.ok) return;
      const saved = await api.uploadDistillFile(file);
      setReport(saved.report);
      setUploads((prev) => [saved.upload, ...prev.filter((u) => u.id !== saved.upload.id)]);
      onSelect(saved.upload);
    } catch (e: unknown) {
      const err = e as Error & { report?: DistillValidateReport };
      if (err.report) setReport(err.report);
      else setError(err.message || "Upload failed");
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await api.deleteDistillUpload(id);
      setUploads((prev) => prev.filter((u) => u.id !== id));
      if (selectedId === id) {
        onSelect(null);
        setReport(null);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  };

  const inspect = async (row: DistillUpload) => {
    onSelect(row);
    try {
      const detail = await api.getDistillUpload(row.id);
      setReport(detail.report);
    } catch {
      setReport(null);
    }
  };

  return (
    <div className="space-y-4">
      <p className="text-sm text-th-text-secondary leading-relaxed max-w-2xl">
        Bring your own traces. We validate the file, keep the valid rows, and use them
        on the next train instead of Otto sessions.
      </p>
      <FormatSpec />

      <label
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          void handleFiles(e.dataTransfer.files);
        }}
        className={`flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-8 cursor-pointer transition-colors ${
          drag
            ? "border-violet-400 bg-violet-500/10"
            : "border-th-border bg-th-card-bg hover:border-violet-500/40 hover:bg-violet-500/5"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".jsonl,.json,application/json"
          className="sr-only"
          onChange={(e) => void handleFiles(e.target.files)}
        />
        {busy ? (
          <Loader2 size={20} className="animate-spin text-violet-400" />
        ) : (
          <Upload size={20} className="text-violet-400" />
        )}
        <p className="text-sm text-th-text-primary">
          {busy ? "Checking format…" : "Drop a .jsonl file here"}
        </p>
        <p className="text-[11px] text-th-text-muted">or click to browse · UTF-8 · 50 MB max</p>
      </label>

      {error && <p className="text-xs text-red-400">{error}</p>}
      {report && <ReportCard report={report} />}

      <div className="bg-th-card-bg border border-th-card-border rounded-xl overflow-hidden">
        <div className="px-4 py-2.5 border-b border-th-border/50 flex items-center justify-between">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-th-text-muted">
            Saved files
          </p>
          {loading && <Loader2 size={12} className="animate-spin text-th-text-muted" />}
        </div>
        {uploads.length === 0 && !loading && (
          <p className="px-4 py-8 text-center text-xs text-th-text-muted">
            No uploads yet. Valid files appear here and can be selected for training.
          </p>
        )}
        <div className="divide-y divide-th-border/20">
          {uploads.map((u) => {
            const selected = selectedId === u.id;
            return (
              <div
                key={u.id}
                className={`flex items-center gap-3 px-4 py-3 ${
                  selected ? "bg-violet-500/10" : ""
                }`}
              >
                <button
                  type="button"
                  onClick={() => void inspect(u)}
                  className="min-w-0 flex-1 text-left"
                >
                  <p className="text-[12px] font-medium text-th-text-primary truncate flex items-center gap-1.5">
                    <FileUp size={12} className="text-violet-400 shrink-0" />
                    {u.filename}
                  </p>
                  <p className="text-[10px] text-th-text-muted mt-0.5">
                    {u.n_valid.toLocaleString()} valid
                    {u.n_invalid ? ` · ${u.n_invalid} skipped` : ""}
                    {" · "}{formatBytes(u.bytes)}
                    {u.uploaded_at ? ` · ${formatWhen(u.uploaded_at)}` : ""}
                  </p>
                </button>
                <button
                  type="button"
                  onClick={() => onSelect(u)}
                  className={`shrink-0 px-2.5 py-1 rounded-md text-[10px] font-medium border ${
                    selected
                      ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                      : "text-th-text-tertiary border-th-border hover:text-th-text-primary"
                  }`}
                >
                  {selected ? "Selected" : "Use for train"}
                </button>
                <button
                  type="button"
                  onClick={() => void handleDelete(u.id)}
                  className="p-1.5 rounded text-th-text-muted hover:text-red-400"
                  title="Delete upload"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
