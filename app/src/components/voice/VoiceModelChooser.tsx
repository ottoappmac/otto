/**
 * VoiceModelChooser — shared component for selecting and downloading STT models.
 *
 * Reused by:
 *   - SetupWizard.tsx (voice step)
 *   - SettingsPage.tsx (Voice section)
 *
 * Mirrors the UX of ModelChooser.tsx: fit badges, download button + progress bar,
 * and polling every 800 ms.
 */

import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import {
  CheckCircle2,
  Download,
  HardDrive,
  Loader2,
  Mic,
  Zap,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import type { VoiceCatalogRow, VoiceConfig } from "../../types";

// Re-use the MLX download job type shape (same backend pattern)
interface DownloadJob {
  job_id: string;
  repo_id: string;
  status: string;
  bytes_done: number;
  bytes_total: number;
  rate_bps: number;
  eta_seconds: number | null;
  current_file: string;
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface VoiceModelChooserProps {
  /** Current voice config — highlights selected models. */
  config: Partial<VoiceConfig>;
  /** Called when the user selects (or finishes downloading) an STT model. */
  onSelectStt?: (repoId: string) => void;
  /** Called when the user selects (or finishes downloading) a wake model. */
  onSelectWake?: (repoId: string) => void;
  /** Only show specific kinds (default: all). */
  kinds?: Array<"stt" | "wake">;
  hfToken?: string;
  cacheDir?: string;
}

// ---------------------------------------------------------------------------
// Fit badge (mirrors mlx ModelChooser)
// ---------------------------------------------------------------------------

function FitBadge({ fits }: { fits: VoiceCatalogRow["fits"] }) {
  const map: Record<string, { label: string; cls: string }> = {
    comfortable: { label: "Fits", cls: "bg-green-500/15 text-green-400 border-green-500/30" },
    tight: { label: "Tight", cls: "bg-yellow-500/15 text-yellow-400 border-yellow-500/30" },
    over: { label: "Too large", cls: "bg-red-500/15 text-red-400 border-red-500/30" },
    unknown: { label: "Unknown", cls: "bg-zinc-500/15 text-zinc-400 border-zinc-500/30" },
  };
  const { label, cls } = map[fits] ?? map.unknown;
  return (
    <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${cls}`}>
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Download progress bar
// ---------------------------------------------------------------------------

function DownloadProgress({ job }: { job: DownloadJob }) {
  const pct =
    job.bytes_total > 0 ? Math.min(100, (job.bytes_done / job.bytes_total) * 100) : 0;

  function bytesToHuman(n: number) {
    if (!n || n < 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let i = 0;
    let v = n;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
  }

  return (
    <div className="mt-2 space-y-1">
      <div className="h-1.5 rounded-full bg-zinc-700 overflow-hidden">
        <div
          className="h-full bg-blue-500 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-zinc-500">
        <span>{bytesToHuman(job.bytes_done)} / {bytesToHuman(job.bytes_total)}</span>
        {job.eta_seconds != null && <span>{job.eta_seconds}s remaining</span>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Kind icon + label
// ---------------------------------------------------------------------------

function KindLabel({ kind }: { kind: "stt" | "wake" }) {
  if (kind === "wake") {
    return (
      <span className="flex items-center gap-1 text-xs font-semibold text-zinc-400 uppercase tracking-wider">
        <Zap className="w-3 h-3" /> Wake Word
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1 text-xs font-semibold text-zinc-400 uppercase tracking-wider">
      <Mic className="w-3 h-3" /> Speech to Text
    </span>
  );
}

// ---------------------------------------------------------------------------
// Single model row
// ---------------------------------------------------------------------------

function ModelRow({
  row,
  isSelected,
  onDownload,
  onUse,
  downloadJob,
}: {
  row: VoiceCatalogRow;
  isSelected: boolean;
  onDownload: (row: VoiceCatalogRow) => void;
  onUse: (row: VoiceCatalogRow) => void;
  downloadJob?: DownloadJob;
}) {
  const isBuiltin = row.repo_id.startsWith("__builtin__");
  const isDownloading = downloadJob && downloadJob.status === "running";
  const isComplete = row.cache_complete || row.already_cached || isBuiltin;

  return (
    <div
      className={`rounded-lg border p-3 transition-colors ${
        isSelected
          ? "border-blue-500/50 bg-blue-500/5"
          : "border-zinc-700 bg-zinc-800/50 hover:border-zinc-600"
      }`}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-medium text-zinc-100 truncate">
              {row.display_name}
            </span>
            {row.featured && (
              <span className="text-[9px] font-semibold text-blue-400 bg-blue-500/10 border border-blue-500/20 px-1.5 py-0.5 rounded">
                Recommended
              </span>
            )}
            {isBuiltin && (
              <span className="text-[9px] font-semibold text-green-400 bg-green-500/10 border border-green-500/20 px-1.5 py-0.5 rounded">
                Built-in
              </span>
            )}
            {isComplete && !isBuiltin && (
              <CheckCircle2 className="w-3.5 h-3.5 text-green-400 flex-shrink-0" />
            )}
          </div>
          {row.blurb && (
            <p className="text-xs text-zinc-400 mt-0.5 leading-snug">{row.blurb}</p>
          )}
          <div className="flex items-center gap-2 mt-1 flex-wrap">
            <FitBadge fits={row.fits} />
            {row.weights_gb > 0 && (
              <span className="flex items-center gap-0.5 text-[10px] text-zinc-500">
                <HardDrive className="w-3 h-3" />
                {row.weights_gb.toFixed(2)} GB
              </span>
            )}
            {row.language && row.language !== "multilingual" && (
              <span className="text-[10px] text-zinc-500">{row.language}</span>
            )}
          </div>
        </div>

        {/* Action button */}
        <div className="flex flex-col items-end gap-1">
          {isBuiltin || isComplete ? (
            <button
              className={`text-xs px-2.5 py-1 rounded font-medium transition-colors ${
                isSelected
                  ? "bg-blue-600 text-white"
                  : "bg-zinc-700 text-zinc-200 hover:bg-zinc-600"
              }`}
              onClick={() => onUse(row)}
            >
              {isSelected ? "Selected" : "Use"}
            </button>
          ) : isDownloading ? (
            <button
              className="text-xs px-2.5 py-1 rounded font-medium bg-zinc-700 text-zinc-400 cursor-default flex items-center gap-1"
              disabled
            >
              <Loader2 className="w-3 h-3 animate-spin" />
              Downloading
            </button>
          ) : (
            <button
              className="flex items-center gap-1 text-xs px-2.5 py-1 rounded font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors disabled:opacity-40"
              onClick={() => onDownload(row)}
              disabled={row.fits === "over" || !row.disk_ok}
            >
              <Download className="w-3 h-3" />
              Download
            </button>
          )}
        </div>
      </div>

      {/* Download progress */}
      {isDownloading && downloadJob && <DownloadProgress job={downloadJob} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function VoiceModelChooser({
  config,
  onSelectStt,
  onSelectWake,
  kinds,
  hfToken = "",
  cacheDir,
}: VoiceModelChooserProps) {
  const [rows, setRows] = useState<VoiceCatalogRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [jobs, setJobs] = useState<Record<string, DownloadJob>>({});

  const loadCatalog = useCallback(async () => {
    try {
      const res = await api.voiceCatalog();
      setRows(res.rows);
    } catch {
      // silently ignore — catalog may not be available yet
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadCatalog();
  }, [loadCatalog]);

  const activeJobIds = useMemo(
    () => Object.values(jobs).filter((j) => j.status === "running").map((j) => j.job_id),
    [jobs],
  );

  const loadCatalogRef = useRef(loadCatalog);
  loadCatalogRef.current = loadCatalog;
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;

  useEffect(() => {
    if (activeJobIds.length === 0) return;
    const id = setInterval(async () => {
      for (const repoId of activeJobIds) {
        const entry = jobsRef.current[repoId];
        if (!entry) continue;
        try {
          const job = await api.mlxDownloadStatus(entry.job_id) as unknown as DownloadJob;
          setJobs((prev) => ({ ...prev, [repoId]: { ...job, repo_id: repoId } }));
          if (job.status === "done") {
            void loadCatalogRef.current();
          }
        } catch {
          setJobs((prev) => {
            const next = { ...prev };
            delete next[repoId];
            return next;
          });
        }
      }
    }, 800);
    return () => clearInterval(id);
  }, [activeJobIds]);

  const handleDownload = useCallback(async (row: VoiceCatalogRow) => {
    try {
      const res = await api.mlxDownload({
        repo_id: row.repo_id,
        label: row.display_name,
        hf_token: hfToken,
        ...(cacheDir ? { cache_dir: cacheDir } : {}),
      });
      const jobId = (res as { job_id: string }).job_id;
      setJobs((prev) => ({
        ...prev,
        [row.repo_id]: { job_id: jobId, repo_id: row.repo_id, status: "running", bytes_done: 0, bytes_total: 0, rate_bps: 0, eta_seconds: null, current_file: "" },
      }));
    } catch (err) {
      console.error("Voice model download failed", err);
    }
  }, [hfToken, cacheDir]);

  const handleUse = useCallback((row: VoiceCatalogRow) => {
    if (row.kind === "stt") onSelectStt?.(row.repo_id);
    else if (row.kind === "wake") onSelectWake?.(row.repo_id);
  }, [onSelectStt, onSelectWake]);

  const visibleKinds: Array<"stt" | "wake"> = kinds ?? ["stt", "wake"];

  const jobByRepoId = Object.values(jobs).reduce<Record<string, DownloadJob>>((acc, j) => {
    acc[j.repo_id] = j;
    return acc;
  }, {});

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8 text-zinc-500">
        <Loader2 className="w-5 h-5 animate-spin mr-2" />
        Loading voice model catalog…
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {visibleKinds.map((kind) => {
        const kindRows = rows.filter((r) => r.kind === kind);
        if (kindRows.length === 0) return null;
        return (
          <div key={kind}>
            <div className="mb-2">
              <KindLabel kind={kind} />
            </div>
            <div className="space-y-2">
              {kindRows.map((row) => {
                const isSelected =
                  kind === "stt"
                    ? config.stt_model === row.repo_id
                    : config.wake_model === row.repo_id;
                return (
                  <ModelRow
                    key={row.repo_id}
                    row={row}
                    isSelected={isSelected}
                    onDownload={handleDownload}
                    onUse={handleUse}
                    downloadJob={jobByRepoId[row.repo_id]}
                  />
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
