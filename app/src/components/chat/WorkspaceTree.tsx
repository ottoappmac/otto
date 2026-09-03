import { useCallback, useEffect, useState } from "react";
import { ChevronDown, ChevronRight, FileText, Folder, FolderOpen } from "lucide-react";
import { api } from "../../hooks/useApi";
import type { WorkspaceTreeEntry } from "../../types";
import { artifactTypeFromPath, type ArtifactType } from "./ArtifactPanel";

interface WorkspaceTreeProps {
  sessionId: string;
  onOpenFile: (relPath: string, fileUrl: string, type: ArtifactType) => void;
  fileUrl: (relPath: string) => string;
}

export default function WorkspaceTree({ sessionId, onOpenFile, fileUrl }: WorkspaceTreeProps) {
  return <TreeLevel sessionId={sessionId} rel="" depth={0} onOpenFile={onOpenFile} fileUrl={fileUrl} />;
}

function TreeLevel({
  sessionId,
  rel,
  depth,
  onOpenFile,
  fileUrl,
}: {
  sessionId: string;
  rel: string;
  depth: number;
  onOpenFile: (relPath: string, fileUrl: string, type: ArtifactType) => void;
  fileUrl: (relPath: string) => string;
}) {
  const [entries, setEntries] = useState<WorkspaceTreeEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;
    api.getWorkspaceTree(sessionId, rel)
      .then((r) => { if (!cancelled) setEntries(r.entries); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : "Failed to list"); });
    return () => { cancelled = true; };
  }, [sessionId, rel]);

  const toggle = useCallback((path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  if (error) {
    return <p className="text-[11px] text-red-400/80 px-2 py-1">{error}</p>;
  }
  if (!entries) {
    return <p className="text-[11px] text-th-text-muted px-2 py-1">Loading…</p>;
  }

  return (
    <div>
      {entries.map((entry) => {
        const indent = { paddingLeft: `${8 + depth * 12}px` };
        if (entry.is_dir) {
          const isOpen = expanded.has(entry.path);
          return (
            <div key={`d:${entry.path}`}>
              <button
                type="button"
                style={indent}
                onClick={() => toggle(entry.path)}
                className="w-full flex items-center gap-1.5 py-0.5 text-xs text-th-text-secondary hover:text-th-text-primary hover:bg-th-surface-hover rounded"
              >
                {isOpen ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
                {isOpen ? <FolderOpen size={12} className="text-sky-400/80" /> : <Folder size={12} className="text-sky-400/70" />}
                <span className="truncate">{entry.name}</span>
              </button>
              {isOpen && (
                <TreeLevel
                  sessionId={sessionId}
                  rel={entry.path}
                  depth={depth + 1}
                  onOpenFile={onOpenFile}
                  fileUrl={fileUrl}
                />
              )}
            </div>
          );
        }
        const type = artifactTypeFromPath(entry.name) ?? "code";
        return (
          <button
            key={`f:${entry.path}`}
            type="button"
            style={indent}
            onClick={() => onOpenFile(entry.path, fileUrl(entry.path), type)}
            className="w-full flex items-center gap-1.5 py-0.5 text-xs text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover rounded"
          >
            <span className="w-[11px]" />
            <FileText size={12} className="text-th-text-muted" />
            <span className="truncate font-mono">{entry.name}</span>
          </button>
        );
      })}
    </div>
  );
}
