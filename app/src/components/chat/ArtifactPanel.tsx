import { useEffect, useRef, useState } from "react";
import { X, ExternalLink, FileText, Globe, RefreshCw, Image as ImageIcon, FileJson } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import mammoth from "mammoth";
import * as XLSX from "xlsx";

export type ArtifactType =
  | "html" | "md" | "pdf" | "docx" | "txt" | "csv" | "xlsx" | "image" | "json";

export interface Artifact {
  path: string;       // virtual path, e.g. "output/report.html"
  fileUrl: string;    // full URL to fetch / display
  type: ArtifactType;
}

/** Map a file path to a viewable artifact type, or null if unknown. */
export function artifactTypeFromPath(path: string): ArtifactType | null {
  const p = path.toLowerCase();
  if (p.endsWith(".html")) return "html";
  if (p.endsWith(".md")) return "md";
  if (p.endsWith(".pdf")) return "pdf";
  if (p.endsWith(".docx")) return "docx";
  if (p.endsWith(".txt")) return "txt";
  if (p.endsWith(".csv")) return "csv";
  if (p.endsWith(".xlsx")) return "xlsx";
  if (p.endsWith(".json")) return "json";
  if (/\.(png|jpe?g|gif|webp|svg|bmp|ico|avif)$/.test(p)) return "image";
  return null;
}

// ---------------------------------------------------------------------------
// Shared prose classes — same style as agent message bubbles
// ---------------------------------------------------------------------------
const PROSE_CLS = `
  prose max-w-[860px] mx-auto break-words
  [font-family:'Source_Serif_4_Variable',Georgia,serif]
  text-th-text-primary
  prose-headings:text-th-text-primary prose-headings:font-semibold prose-headings:tracking-tight
  prose-headings:[font-family:'Inter_Variable',ui-sans-serif,system-ui,sans-serif]
  prose-h1:text-2xl prose-h1:mt-10 prose-h1:mb-5 prose-h1:leading-snug
  prose-h2:text-[19px] prose-h2:mt-8 prose-h2:mb-4 prose-h2:leading-snug
  prose-h3:text-[17px] prose-h3:mt-7 prose-h3:mb-3 prose-h3:leading-snug
  prose-p:text-[16px] prose-p:leading-[1.85] prose-p:my-5
  prose-strong:text-th-text-primary prose-strong:font-semibold
  prose-li:text-[16px] prose-li:leading-[1.8] prose-li:my-2
  prose-ul:my-5 prose-ol:my-5 prose-ul:pl-6 prose-ol:pl-6
  prose-blockquote:border-l-[3px] prose-blockquote:border-th-border-strong
  prose-blockquote:text-th-text-secondary prose-blockquote:not-italic prose-blockquote:pl-5 prose-blockquote:my-6
  prose-code:text-th-text-secondary prose-code:bg-th-code-bg prose-code:px-1.5 prose-code:py-0.5
  prose-code:rounded-md prose-code:text-[13px] prose-code:[font-family:ui-monospace,SFMono-Regular,Menlo,monospace]
  prose-code:font-normal prose-code:before:content-none prose-code:after:content-none
  prose-pre:bg-th-code-bg prose-pre:border prose-pre:border-th-border prose-pre:rounded-xl prose-pre:my-6 prose-pre:p-5
  prose-pre:text-[13px] prose-pre:[font-family:ui-monospace,SFMono-Regular,Menlo,monospace]
  prose-table:text-[14px] prose-table:border-collapse
  prose-th:text-th-text-secondary prose-th:font-semibold prose-th:py-3 prose-th:px-4 prose-th:border-b prose-th:border-th-border
  prose-th:[font-family:'Inter_Variable',ui-sans-serif,system-ui,sans-serif]
  prose-td:py-3 prose-td:px-4 prose-td:border-b prose-td:border-th-border/50
  prose-hr:border-th-border prose-hr:my-8
  prose-a:text-blue-400 prose-a:no-underline hover:prose-a:underline
`.replace(/\n\s*/g, " ");

const mdComponents: Components = {
  table: ({ children }) => (
    <div className="overflow-x-auto my-4 rounded-xl border border-th-border">
      <table className="w-full border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-th-inset-bg">{children}</thead>,
  tr: ({ children }) => <tr className="border-b border-th-border/50 last:border-0">{children}</tr>,
};

// ---------------------------------------------------------------------------
// MdViewerModal — full-screen centred modal for .md files
// ---------------------------------------------------------------------------

interface MdViewerModalProps {
  artifact: Artifact;
  onClose: () => void;
}

export function MdViewerModal({ artifact, onClose }: MdViewerModalProps) {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const backdropRef = useRef<HTMLDivElement>(null);

  const filename = artifact.path.split("/").pop() ?? artifact.path;

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch(artifact.fileUrl)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); })
      .then(setContent)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [artifact.fileUrl]);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div
      ref={backdropRef}
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 backdrop-blur-sm p-6 overflow-y-auto"
      onClick={(e) => { if (e.target === backdropRef.current) onClose(); }}
    >
      <div className="relative w-full max-w-5xl bg-th-bg-secondary border border-th-border rounded-2xl shadow-2xl my-6 flex flex-col">
        {/* Header */}
        <div className="flex items-center gap-2.5 px-6 py-4 border-b border-th-border shrink-0 sticky top-0 bg-th-bg-secondary rounded-t-2xl z-10">
          <FileText size={15} className="text-emerald-400 shrink-0" />
          <span className="text-sm font-medium text-th-text-primary flex-1 truncate" title={artifact.path}>
            {filename}
          </span>
          <div className="flex items-center gap-1 shrink-0">
            <a
              href={artifact.fileUrl}
              target="_blank"
              rel="noopener noreferrer"
              title="Open raw file"
              className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-colors"
            >
              <ExternalLink size={14} />
            </a>
            <button
              onClick={onClose}
              title="Close (Esc)"
              className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-colors"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="px-8 py-8">
          {loading ? (
            <div className="flex items-center justify-center py-24 text-sm text-th-text-muted">Loading…</div>
          ) : error ? (
            <div className="flex items-center justify-center py-24 text-sm text-red-400">{error}</div>
          ) : (
            <div className={PROSE_CLS}>
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
                {content ?? ""}
              </ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface ArtifactPanelProps {
  artifact: Artifact;
  onClose: () => void;
}

// Shared table style for CSV/XLSX
const TABLE_CLS = "w-full text-xs border-collapse";
const TH_CLS = "text-left px-3 py-2 font-semibold text-th-text-secondary bg-th-inset-bg border border-th-border sticky top-0";
const TD_CLS = "px-3 py-1.5 border border-th-border/60 text-th-text-primary align-top";

function SpreadsheetViewer({ rows }: { rows: string[][] }) {
  if (rows.length === 0) return <p className="text-sm text-th-text-muted p-4">Empty file</p>;
  const [header, ...body] = rows;
  return (
    <div className="h-full overflow-auto">
      <table className={TABLE_CLS}>
        <thead>
          <tr>{header.map((cell, i) => <th key={i} className={TH_CLS}>{cell}</th>)}</tr>
        </thead>
        <tbody>
          {body.map((row, ri) => (
            <tr key={ri} className="odd:bg-th-bg-secondary even:bg-th-card-bg hover:bg-th-surface-hover transition-colors">
              {header.map((_, ci) => <td key={ci} className={TD_CLS}>{row[ci] ?? ""}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ArtifactPanel({ artifact, onClose }: ArtifactPanelProps) {
  const [mdContent, setMdContent] = useState<string | null>(null);
  const [docxHtml, setDocxHtml] = useState<string | null>(null);
  const [plainText, setPlainText] = useState<string | null>(null);
  const [jsonText, setJsonText] = useState<string | null>(null);
  const [sheetRows, setSheetRows] = useState<string[][] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [iframeKey, setIframeKey] = useState(0);

  useEffect(() => {
    setMdContent(null);
    setDocxHtml(null);
    setPlainText(null);
    setJsonText(null);
    setSheetRows(null);
    setError(null);

    // Guard against a slow fetch for a previous artifact resolving after the
    // user switched artifacts (or closed the panel) and clobbering state.
    let cancelled = false;
    const apply = <T,>(setter: (v: T) => void) => (v: T) => { if (!cancelled) setter(v); };
    const applyError = (e: unknown) => { if (!cancelled) setError(String(e)); };
    const applyDone = () => { if (!cancelled) setLoading(false); };

    if (artifact.type === "json") {
      setLoading(true);
      fetch(artifact.fileUrl)
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); })
        .then(apply((text: string) => {
          try { setJsonText(JSON.stringify(JSON.parse(text), null, 2)); }
          catch { setJsonText(text); }
        }))
        .catch(applyError)
        .finally(applyDone);
    } else if (artifact.type === "md") {
      setLoading(true);
      fetch(artifact.fileUrl)
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); })
        .then(apply(setMdContent))
        .catch(applyError)
        .finally(applyDone);
    } else if (artifact.type === "txt") {
      setLoading(true);
      fetch(artifact.fileUrl)
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); })
        .then(apply(setPlainText))
        .catch(applyError)
        .finally(applyDone);
    } else if (artifact.type === "csv") {
      setLoading(true);
      fetch(artifact.fileUrl)
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); })
        .then(apply((text: string) => {
          const wb = XLSX.read(text, { type: "string" });
          const ws = wb.Sheets[wb.SheetNames[0]];
          setSheetRows(XLSX.utils.sheet_to_json<string[]>(ws, { header: 1 }) as string[][]);
        }))
        .catch(applyError)
        .finally(applyDone);
    } else if (artifact.type === "xlsx") {
      setLoading(true);
      fetch(artifact.fileUrl)
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); })
        .then(apply((buf: ArrayBuffer) => {
          const wb = XLSX.read(buf, { type: "array" });
          const ws = wb.Sheets[wb.SheetNames[0]];
          setSheetRows(XLSX.utils.sheet_to_json<string[]>(ws, { header: 1 }) as string[][]);
        }))
        .catch(applyError)
        .finally(applyDone);
    } else if (artifact.type === "docx") {
      setLoading(true);
      fetch(artifact.fileUrl)
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); })
        .then((buf) => mammoth.convertToHtml({ arrayBuffer: buf }))
        .then(apply((result: { value: string }) => setDocxHtml(result.value)))
        .catch(applyError)
        .finally(applyDone);
    }

    return () => { cancelled = true; };
  }, [artifact.fileUrl, artifact.type]);

  const filename = artifact.path.split("/").pop() ?? artifact.path;
  const iconColor =
    artifact.type === "html" ? "text-blue-400" :
    artifact.type === "pdf"  ? "text-red-400"  :
    artifact.type === "xlsx" || artifact.type === "csv" ? "text-emerald-400" :
    artifact.type === "image" ? "text-purple-400" :
    artifact.type === "json" ? "text-amber-400" :
    "text-th-text-secondary";
  const Icon =
    artifact.type === "html" ? Globe :
    artifact.type === "image" ? ImageIcon :
    artifact.type === "json" ? FileJson :
    FileText;

  return (
    <div className="flex flex-col h-full border-l border-th-border bg-th-bg-secondary min-w-0">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-th-border shrink-0">
        <Icon size={14} className={`${iconColor} shrink-0`} />
        <span className="text-sm font-medium text-th-text-primary truncate flex-1" title={artifact.path}>
          {filename}
        </span>
        <div className="flex items-center gap-1 shrink-0">
          {(artifact.type === "html" || artifact.type === "pdf") && (
            <button
              onClick={() => setIframeKey((k) => k + 1)}
              title="Reload"
              className="p-1.5 rounded-md text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-colors"
            >
              <RefreshCw size={13} />
            </button>
          )}
          {!artifact.fileUrl.startsWith("data:") && (
            <a
              href={artifact.fileUrl}
              target="_blank"
              rel="noopener noreferrer"
              title="Open in new tab"
              className="p-1.5 rounded-md text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-colors"
            >
              <ExternalLink size={13} />
            </a>
          )}
          <button
            onClick={onClose}
            title="Close"
            className="p-1.5 rounded-md text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-colors"
          >
            <X size={13} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {artifact.type === "html" ? (
          <iframe
            key={iframeKey}
            src={artifact.fileUrl}
            className="w-full h-full bg-white"
            title={filename}
            // No allow-same-origin: combined with allow-scripts it would let
            // agent-generated (or web-derived) HTML escape the sandbox and
            // reach the app origin's cookies/localStorage/API.
            sandbox="allow-scripts allow-forms"
          />
        ) : artifact.type === "pdf" ? (
          <iframe
            key={iframeKey}
            src={artifact.fileUrl}
            className="w-full h-full bg-white"
            title={filename}
          />
        ) : artifact.type === "image" ? (
          <div className="h-full overflow-auto flex items-center justify-center p-6 bg-th-inset-bg/40">
            <img
              src={artifact.fileUrl}
              alt={filename}
              className="max-w-full max-h-full object-contain rounded-lg border border-th-border bg-white"
            />
          </div>
        ) : loading ? (
          <div className="flex items-center justify-center h-full text-sm text-th-text-muted">Loading…</div>
        ) : error ? (
          <div className="flex items-center justify-center h-full text-sm text-red-400 px-6 text-center">
            Failed to load: {error}
          </div>
        ) : artifact.type === "docx" ? (
          <div className="h-full overflow-y-auto px-6 py-5">
            <div
              className="docx-content prose prose-sm max-w-none text-th-text-primary [&_table]:border-collapse [&_td]:border [&_td]:border-th-border [&_td]:px-3 [&_td]:py-1.5 [&_th]:border [&_th]:border-th-border [&_th]:px-3 [&_th]:py-1.5 [&_th]:bg-th-inset-bg [&_a]:text-blue-400 [&_a]:underline"
              dangerouslySetInnerHTML={{ __html: docxHtml ?? "" }}
            />
          </div>
        ) : (artifact.type === "csv" || artifact.type === "xlsx") && sheetRows ? (
          <SpreadsheetViewer rows={sheetRows} />
        ) : artifact.type === "txt" ? (
          <div className="h-full overflow-y-auto px-6 py-5">
            <pre className="text-xs text-th-text-primary font-mono whitespace-pre-wrap break-words leading-relaxed">
              {plainText ?? ""}
            </pre>
          </div>
        ) : artifact.type === "json" ? (
          <div className="h-full overflow-auto px-6 py-5">
            <pre className="text-xs text-th-text-primary font-mono whitespace-pre leading-relaxed">
              {jsonText ?? ""}
            </pre>
          </div>
        ) : (
          <div className="h-full overflow-y-auto px-6 py-5">
            <div className="prose prose-sm max-w-none text-th-text-primary [&_pre]:bg-th-code-bg [&_pre]:border [&_pre]:border-th-border [&_pre]:rounded-lg [&_pre]:whitespace-pre-wrap [&_code]:text-th-text-secondary [&_a]:text-blue-400 [&_a]:underline">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {mdContent ?? ""}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
