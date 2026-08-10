/**
 * SessionFileTree — the session's files as a collapsible folder tree.
 *
 * The backend returns a flat, fully-recursed list of file paths (no directory
 * entries), so the hierarchy is reconstructed here by splitting on "/".
 * Folders start collapsed, which keeps output-heavy sessions readable — a
 * single video analysis can drop a hundred frames into `video-frames/`.
 */

import { useMemo, useState } from "react";
import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Download,
  FileJson,
  FileText,
  Folder,
  FolderOpen,
  Image as ImageIcon,
  Trash2,
  Video,
} from "lucide-react";
import { artifactTypeFromPath, type ArtifactType } from "./ArtifactPanel";
import { formatFileSize } from "../../utils/formatFileSize";
import { buildFileTree, type FileTreeNode, type SessionFileEntry } from "../../utils/fileTree";

interface SessionFileTreeProps {
  files: SessionFileEntry[];
  fileUrl: (path: string) => string;
  downloadedFile: string | null;
  onDownload: (path: string) => void;
  onDelete: (path: string) => void;
  onReveal: (path: string) => void;
  onOpenArtifact: (path: string, url: string, type: ArtifactType) => void;
}

function iconFor(type: ArtifactType | null) {
  const cls =
    type === "pdf" ? "text-red-400/70 group-hover:text-red-400"
    : type === "csv" || type === "xlsx" ? "text-emerald-400/70 group-hover:text-emerald-400"
    : type === "image" ? "text-purple-400/70 group-hover:text-purple-400"
    : type === "video" ? "text-rose-400/70 group-hover:text-rose-400"
    : type === "json" ? "text-amber-400/70 group-hover:text-amber-400"
    : type ? "text-blue-400/70 group-hover:text-blue-400"
    : "text-th-text-muted group-hover:text-th-text-secondary";
  const Icon =
    type === "image" ? ImageIcon
    : type === "video" ? Video
    : type === "json" ? FileJson
    : FileText;
  return { Icon, cls };
}

export default function SessionFileTree({
  files,
  fileUrl,
  downloadedFile,
  onDownload,
  onDelete,
  onReveal,
  onOpenArtifact,
}: SessionFileTreeProps) {
  const tree = useMemo(() => buildFileTree(files), [files]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggle = (path: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  const renderNode = (node: FileTreeNode, depth: number): React.ReactNode => {
    const indent = { paddingLeft: `${12 + depth * 14}px` };

    if (!node.file) {
      const isOpen = expanded.has(node.path);
      return (
        <div key={`dir:${node.path}`}>
          <div
            role="button"
            tabIndex={0}
            onClick={() => toggle(node.path)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                toggle(node.path);
              }
            }}
            style={indent}
            className="flex items-center gap-2.5 pr-3 py-1.5 rounded-lg hover:bg-th-surface-hover transition-colors group cursor-pointer"
          >
            {isOpen ? (
              <ChevronDown size={12} className="shrink-0 text-th-text-muted" />
            ) : (
              <ChevronRight size={12} className="shrink-0 text-th-text-muted" />
            )}
            {isOpen ? (
              <FolderOpen size={13} className="shrink-0 text-sky-400/70 group-hover:text-sky-400" />
            ) : (
              <Folder size={13} className="shrink-0 text-sky-400/70 group-hover:text-sky-400" />
            )}
            <span className="text-xs text-th-text-secondary truncate flex-1 font-mono">
              {node.name}
            </span>
            <span className="text-[10px] text-th-text-muted shrink-0">
              {node.count} {node.count === 1 ? "file" : "files"}
            </span>
            <span className="text-[10px] text-th-text-muted shrink-0">
              {formatFileSize(node.size)}
            </span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onReveal(node.path);
              }}
              className="text-th-text-muted hover:text-th-text-secondary transition-colors shrink-0"
              title="Show in folder"
            >
              <FolderOpen size={12} />
            </button>
          </div>
          {isOpen && node.children.map((c) => renderNode(c, depth + 1))}
        </div>
      );
    }

    const file = node.file;
    const type = artifactTypeFromPath(file.path);
    const { Icon, cls } = iconFor(type);
    return (
      <div
        key={`file:${file.path}`}
        style={indent}
        className={`flex items-center gap-2.5 pr-3 py-1.5 rounded-lg hover:bg-th-surface-hover transition-colors group ${
          type ? "cursor-pointer" : "cursor-default"
        }`}
        onClick={
          type ? () => onOpenArtifact(file.path, fileUrl(file.path), type) : undefined
        }
      >
        <Icon size={13} className={`shrink-0 ${cls}`} />
        <span className="text-xs text-th-text-secondary truncate flex-1 font-mono">
          {node.name}
        </span>
        <span className="text-[10px] text-th-text-muted shrink-0">
          {formatFileSize(file.size)}
        </span>
        {type && (
          <span className="text-[10px] text-th-text-muted group-hover:text-blue-400 transition-colors shrink-0 font-medium">
            Open ↗
          </span>
        )}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onReveal(file.path);
          }}
          className="text-th-text-muted hover:text-th-text-secondary transition-colors shrink-0"
          title="Show in folder"
        >
          <FolderOpen size={12} />
        </button>
        {downloadedFile === file.path ? (
          <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400 font-medium shrink-0">
            <CheckCircle2 size={12} /> Downloaded
          </span>
        ) : (
          <a
            href={fileUrl(file.path)}
            download
            onClick={(e) => {
              e.stopPropagation();
              onDownload(file.path);
            }}
            className="text-th-text-muted hover:text-th-text-secondary transition-colors shrink-0"
            title="Download"
          >
            <Download size={12} />
          </a>
        )}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete(file.path);
          }}
          className="text-th-text-muted hover:text-red-400 transition-colors shrink-0"
          title="Delete"
        >
          <Trash2 size={12} />
        </button>
      </div>
    );
  };

  return <>{tree.map((n) => renderNode(n, 0))}</>;
}
