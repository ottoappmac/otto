/**
 * OutputFileGrid — the "Output files" chip grid shown under the last message.
 *
 * Files sitting in a subdirectory collapse into a single folder chip that
 * expands in place, so a run that emits a hundred sampled video frames shows
 * as one chip rather than a hundred.
 */

import { Fragment, useMemo, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  ExternalLink,
  FileJson,
  FileText,
  Folder,
  FolderOpen,
  Image as ImageIcon,
  Video as VideoIcon,
} from "lucide-react";
import { artifactTypeFromPath, type ArtifactType } from "./ArtifactPanel";
import { buildFileTree, type FileTreeNode, type SessionFileEntry } from "../../utils/fileTree";

interface OutputFileGridProps {
  files: SessionFileEntry[];
  fileUrl: (path: string) => string;
  onOpenArtifact: (path: string, url: string, type: ArtifactType) => void;
}

export default function OutputFileGrid({
  files,
  fileUrl,
  onOpenArtifact,
}: OutputFileGridProps) {
  const tree = useMemo(() => buildFileTree(files), [files]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggle = (path: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  const renderLevel = (nodes: FileTreeNode[]): React.ReactNode => (
    <div className="grid grid-cols-5 gap-2">
      {nodes.map((node) => {
        if (!node.file) {
          const isOpen = expanded.has(node.path);
          return (
            <Fragment key={`dir:${node.path}`}>
              <button
                onClick={() => toggle(node.path)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-th-border bg-th-bg-secondary hover:bg-th-surface-hover hover:border-th-border-strong transition-colors cursor-pointer group text-left min-w-0"
                title={`${node.path} — ${node.count} ${node.count === 1 ? "file" : "files"}`}
              >
                {isOpen ? (
                  <FolderOpen size={12} className="shrink-0 text-sky-400" />
                ) : (
                  <Folder size={12} className="shrink-0 text-sky-400" />
                )}
                <span className="text-xs font-medium text-th-text-secondary group-hover:text-th-text-primary transition-colors truncate flex-1">
                  {node.name}
                </span>
                <span className="text-[10px] text-th-text-muted shrink-0 tabular-nums">
                  {node.count}
                </span>
                {isOpen ? (
                  <ChevronDown size={10} className="text-th-text-muted shrink-0" />
                ) : (
                  <ChevronRight size={10} className="text-th-text-muted shrink-0" />
                )}
              </button>
              {isOpen && (
                <div className="col-span-full pl-3 ml-1 border-l border-th-border">
                  {renderLevel(node.children)}
                </div>
              )}
            </Fragment>
          );
        }

        const type = artifactTypeFromPath(node.file.path);
        if (!type) return null;
        const iconCls =
          type === "pdf" ? "text-red-400"
          : type === "csv" || type === "xlsx" ? "text-emerald-400"
          : type === "image" ? "text-purple-400"
          : type === "video" ? "text-rose-400"
          : type === "json" ? "text-amber-400"
          : "text-blue-400";
        const Icon =
          type === "image" ? ImageIcon
          : type === "video" ? VideoIcon
          : type === "json" ? FileJson
          : FileText;
        const path = node.file.path;
        return (
          <button
            key={`file:${path}`}
            onClick={() => onOpenArtifact(path, fileUrl(path), type)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-th-border bg-th-bg-secondary hover:bg-th-surface-hover hover:border-th-border-strong transition-colors cursor-pointer group text-left min-w-0"
            title={path}
          >
            <Icon size={12} className={`shrink-0 ${iconCls}`} />
            <span className="text-xs font-medium text-th-text-secondary group-hover:text-th-text-primary transition-colors truncate flex-1">
              {node.name}
            </span>
            <ExternalLink size={10} className="text-th-text-muted group-hover:text-blue-400 transition-colors shrink-0" />
          </button>
        );
      })}
    </div>
  );

  return renderLevel(tree);
}
