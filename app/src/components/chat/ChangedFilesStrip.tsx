import { useMemo } from "react";
import { FilePen } from "lucide-react";
import { artifactTypeFromPath, type ArtifactType } from "./ArtifactPanel";
import type { ChatMessage } from "../../types";

export function filePathFromToolArgs(args: Record<string, unknown> | undefined): string | null {
  if (!args) return null;
  const p = args.path ?? args.file_path;
  return typeof p === "string" && p ? p : null;
}

export function collectChangedFiles(messages: ChatMessage[]): { path: string; tool: string }[] {
  const seen = new Map<string, string>();
  for (const m of messages) {
    if (m.type !== "tool_call" && m.type !== "tool_result") continue;
    const tool = m.content;
    if (tool !== "edit_file" && tool !== "write_file" && tool !== "edit") continue;
    const path = filePathFromToolArgs(m.metadata?.args as Record<string, unknown> | undefined);
    if (!path) continue;
    seen.set(path, tool);
  }
  return [...seen.entries()].map(([path, tool]) => ({ path, tool }));
}

interface ChangedFilesStripProps {
  files: { path: string; tool: string }[];
  onOpen: (path: string, fileUrl: string, type: ArtifactType) => void;
  fileUrl: (path: string) => string;
}

export function ChangedFilesStrip({ files, onOpen, fileUrl }: ChangedFilesStripProps) {
  const items = useMemo(() => files.slice(0, 24), [files]);
  if (items.length === 0) return null;

  return (
    <div className="mt-3 ml-10 p-3 rounded-xl border border-th-border bg-th-card-bg max-w-2xl">
      <p className="text-[10px] uppercase tracking-wider font-semibold text-th-text-muted mb-2">
        Changed files ({files.length})
      </p>
      <div className="flex flex-wrap gap-1.5">
        {items.map(({ path, tool }) => {
          const name = path.split("/").pop() ?? path;
          const type = artifactTypeFromPath(path) ?? "code";
          return (
            <button
              key={path}
              type="button"
              onClick={() => onOpen(path, fileUrl(path), type)}
              className="inline-flex items-center gap-1.5 px-2 py-1 rounded-lg border border-th-border bg-th-inset-bg hover:border-blue-400/40 hover:bg-th-surface-hover text-xs text-th-text-secondary transition-colors"
              title={`${tool} ${path}`}
            >
              <FilePen size={11} className="text-emerald-400 shrink-0" />
              <span className="font-mono truncate max-w-[180px] text-th-text-primary">{name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
