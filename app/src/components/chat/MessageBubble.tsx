import { memo, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  ChevronUp, ChevronRight, ChevronDown, Bot,
  Pencil, Check, CheckCheck, X, ShieldAlert, ShieldCheck, ShieldX, Terminal, Image,
  MessageCircleQuestion, Send, Circle, Loader2, CheckCircle2, XCircle,
  AlertTriangle, KeyRound, ExternalLink, Globe, FileText, MessageSquarePlus,
  ScrollText, Folder, Link2,
} from "lucide-react";
import { getToolIcon } from "../../utils/entityIcons";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import type { ChatMessage, MlxStats } from "../../types";
import {
  screenHighRiskCommand,
  describeHighRiskLabel,
} from "../../utils/highRiskCommands";
import { api } from "../../hooks/useApi";
import { getToolLabel } from "../../utils/toolLabels";
import { artifactTypeFromPath } from "./ArtifactPanel";
import type { ArtifactType } from "./ArtifactPanel";

interface MessageBubbleProps {
  message: ChatMessage;
  isThought?: boolean;
  isLatestTodo?: boolean;
  canEdit?: boolean;
  inGroup?: boolean;
  sessionId?: string;
  onEdit?: (content: string) => void;
  onHitlDecision?: (decisions: Array<Record<string, unknown>>) => void;
  onApproveAllSession?: (decisions: Array<Record<string, unknown>>) => void;
  onOpenArtifact?: (path: string, fileUrl: string, type: ArtifactType) => void;
}

// ---------------------------------------------------------------------------
// URL linkification helpers
// ---------------------------------------------------------------------------

// Matches bare domains with optional path that aren't already inside a markdown
// link [...](...) or a code span. Conservative TLD list to avoid false positives
// on file extensions, version strings, etc.
const BARE_URL_RE =
  /(?<!\]\()(?<!\()(?<![`/])(?:https?:\/\/[^\s)"'<>\]]+|(?:[a-zA-Z0-9][-a-zA-Z0-9]*\.)+(?:com|net|org|io|co|au|nz|uk|us|de|fr|app|dev|ai|me|tv|melbourne|sydney|london|tokyo|nyc|berlin)(?:\/[^\s)"'<>\]]*)?)/g;

/**
 * Pre-process agent content so that bare URLs and bare domain names are
 * converted to markdown link syntax before being handed to ReactMarkdown.
 * Code fences and inline code spans are left untouched.
 */
function linkifyContent(content: string): string {
  // Split on fenced code blocks and inline code spans so we never mutate code.
  const segments = content.split(/(```[\s\S]*?```|`[^`\n]+`)/g);
  return segments
    .map((seg, i) => {
      if (i % 2 === 1) return seg; // inside code — leave as-is
      return seg.replace(BARE_URL_RE, (match) => {
        const href = match.startsWith("http") ? match : `https://${match}`;
        return `[${match}](${href})`;
      });
    })
    .join("");
}

/** Shared ReactMarkdown components: links always open in a new tab. */
const markdownComponents: Components = {
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-blue-400 no-underline hover:underline underline-offset-2 transition-colors [overflow-wrap:anywhere] break-all"
    >
      {children}
      <ExternalLink size={10} className="shrink-0 opacity-50 ml-0.5 inline" />
    </a>
  ),
  table: ({ children }) => (
    <div className="overflow-x-auto my-4 rounded-xl border border-th-border">
      <table className="w-full border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-th-inset-bg">{children}</thead>
  ),
  tr: ({ children }) => (
    <tr className="border-b border-th-border/50 last:border-0">{children}</tr>
  ),
};

/**
 * Parse the bracketed attachment prefix lines that ``ChatPage`` prepends to a
 * sent user message (``[Uploaded files: ...]``, ``[Context folders: ...]``,
 * ``[URLs: ...]``) and return them separately from the remaining message text.
 *
 * The prefixes are only ever *prepended* (one per line, before the typed
 * text), so matching is anchored to the leading lines — a user who literally
 * types "[Uploaded files: x]" mid-message keeps their text intact.
 * Limitation: paths containing commas split into separate chips.
 */
function parseUserAttachments(content: string): {
  files: string[];
  folders: string[];
  urls: string[];
  text: string;
} {
  let text = content;
  const files: string[] = [];
  const folders: string[] = [];
  const urls: string[] = [];

  const grab = (re: RegExp, out: string[]) => {
    const m = text.match(re);
    if (m) {
      m[1].split(",").map((s) => s.trim()).filter(Boolean).forEach((v) => out.push(v));
      text = text.slice(m[0].length).replace(/^\s+/, "");
    }
  };

  // Anchored to the start; each grab strips its line so the next prefix
  // (if present) becomes leading again.
  grab(/^\[Uploaded files:\s*([^\]]*)\]/, files);
  grab(/^\[Context folders:\s*([^\]]*)\]/, folders);
  grab(/^\[URLs:\s*([^\]]*)\]/, urls);

  return { files, folders, urls, text };
}

export const MessageBubble = memo(function MessageBubble({ message, isThought, isLatestTodo, canEdit, inGroup, sessionId, onEdit, onHitlDecision, onApproveAllSession, onOpenArtifact }: MessageBubbleProps) {
  const [expanded, setExpanded] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState("");
  const editRef = useRef<HTMLTextAreaElement>(null);
  const [hitlEditing, setHitlEditing] = useState(false);
  const [hitlEditCommand, setHitlEditCommand] = useState("");

  if (message.type === "user") {
    const isContext = Boolean(message.metadata?.isContext);
    const isPending = Boolean(message.metadata?.pending);

    if (isContext) {
      return (
        <div className="flex justify-end">
          <div className="relative max-w-[75%] min-w-0">
            <div className={`bg-violet-500/5 border border-violet-500/20 rounded-2xl rounded-br-sm px-4 py-3 transition-opacity duration-300 ${isPending ? "opacity-70" : "opacity-100"}`}>
              <div className="flex items-center gap-1.5 mb-1.5">
                <MessageSquarePlus size={11} className="text-violet-400/70 shrink-0" />
                <span className="text-[10px] text-violet-400/70 font-medium uppercase tracking-wide">
                  {isPending ? "Context queued…" : "Context added"}
                </span>
              </div>
              <div className="text-sm text-th-text-primary whitespace-pre-wrap break-words leading-relaxed">
                {message.content}
              </div>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="flex justify-end group/msg">
        <div className="relative max-w-[75%] min-w-0">
          {canEdit && !editing && (
            <button
              onClick={() => { setEditText(message.content); setEditing(true); setTimeout(() => editRef.current?.focus(), 50); }}
              className="absolute -left-8 top-2.5 opacity-0 group-hover/msg:opacity-100 transition-opacity text-th-text-muted hover:text-th-text-primary"
              title="Edit & resend"
            >
              <Pencil size={13} />
            </button>
          )}
          <div className="bg-th-surface-hover border border-th-border rounded-2xl rounded-br-sm px-4 py-3">
            {editing ? (
              <div className="space-y-2">
                <textarea
                  ref={editRef}
                  className="w-full bg-th-input-bg border border-th-input-border rounded-lg px-3 py-2 text-sm text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-500/50 resize-none min-h-[60px]"
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (editText.trim()) { onEdit?.(editText.trim()); setEditing(false); } }
                    if (e.key === "Escape") setEditing(false);
                  }}
                />
                <div className="flex gap-2 justify-end">
                  <button
                    onClick={() => setEditing(false)}
                    className="px-2.5 py-1 text-xs text-th-text-tertiary hover:text-th-text-primary rounded-md hover:bg-th-surface-hover transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => { if (editText.trim()) { onEdit?.(editText.trim()); setEditing(false); } }}
                    disabled={!editText.trim()}
                    className="px-2.5 py-1 text-xs text-white bg-blue-600 hover:bg-blue-500 rounded-md transition-colors flex items-center gap-1 disabled:opacity-40"
                  >
                    <Check size={11} /> Save & Resend
                  </button>
                </div>
              </div>
            ) : (() => {
              const { files, folders, urls, text } = parseUserAttachments(message.content);
              const hasAttachments = files.length > 0 || folders.length > 0 || urls.length > 0;
              return (
                <>
                  {hasAttachments && (
                    <div className="flex flex-wrap gap-1.5 mb-2">
                      {files.map((path, i) => {
                        const name = path.replace(/\\/g, "/").split("/").pop() ?? path;
                        const type = artifactTypeFromPath(path);
                        const fileUrl =
                          type && sessionId
                            ? api.getSessionFileUrl(sessionId, path.replace(/^\//, ""))
                            : null;
                        const Icon =
                          type === "html" ? Globe
                          : type === "pdf" ? FileText
                          : type === "image" ? Image
                          : FileText;
                        const iconColor =
                          type === "html" ? "text-blue-400"
                          : type === "pdf" ? "text-red-400"
                          : type === "image" ? "text-purple-400"
                          : "text-emerald-400";
                        const clickable = Boolean(type && fileUrl && onOpenArtifact);
                        return (
                          <button
                            key={`upload-${path}-${i}`}
                            disabled={!clickable}
                            onClick={() => clickable && onOpenArtifact!(path, fileUrl!, type!)}
                            className={`flex items-center gap-2 px-2.5 py-1.5 rounded-xl border border-th-border bg-th-card-bg text-left ${clickable ? "hover:bg-th-surface-hover cursor-pointer group/att" : "cursor-default"}`}
                            title={path}
                          >
                            <Icon size={14} className={`${iconColor} shrink-0`} />
                            <span className="text-xs font-medium text-th-text-primary truncate max-w-[200px]">{name}</span>
                            {clickable && (
                              <span className="text-[10px] text-th-text-muted group-hover/att:text-blue-400 transition-colors shrink-0">Open ↗</span>
                            )}
                          </button>
                        );
                      })}
                      {folders.map((path, i) => {
                        const name = path.replace(/\\/g, "/").split("/").pop() ?? path;
                        return (
                          <div
                            key={`folder-${path}-${i}`}
                            className="flex items-center gap-2 px-2.5 py-1.5 rounded-xl border border-blue-500/20 bg-blue-500/10"
                            title={path}
                          >
                            <Folder size={14} className="text-blue-400 shrink-0" />
                            <span className="text-xs font-medium text-blue-400 truncate max-w-[200px]">{name}</span>
                          </div>
                        );
                      })}
                      {urls.map((url, i) => (
                        <a
                          key={`url-${url}-${i}`}
                          href={url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-2 px-2.5 py-1.5 rounded-xl border border-th-border bg-th-card-bg hover:bg-th-surface-hover"
                          title={url}
                        >
                          <Link2 size={14} className="text-blue-400 shrink-0" />
                          <span className="text-xs font-medium text-th-text-primary truncate max-w-[200px]">{url.replace(/^https?:\/\//, "")}</span>
                        </a>
                      ))}
                    </div>
                  )}
                  {text && (
                    <div className="text-sm text-th-text-primary whitespace-pre-wrap break-words leading-relaxed prose prose-sm max-w-none [&_p]:mb-0 [&_a]:text-blue-400 [&_a]:underline [&_a]:underline-offset-2">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{linkifyContent(text)}</ReactMarkdown>
                    </div>
                  )}
                </>
              );
            })()}
          </div>
        </div>
      </div>
    );
  }

  if (message.type === "agent") {
    const subagentName = message.metadata?.subagent as string | undefined;

    if (subagentName) {
      return (
        <div className={inGroup ? "" : "ml-16 pl-3"}>
          <div className="py-1">
            <div className="text-xs text-th-text-tertiary prose prose-sm max-w-none leading-relaxed break-words [&_p]:mb-1 [&_p:last-child]:mb-0 [&_pre]:bg-th-code-bg [&_pre]:border [&_pre]:border-th-border [&_pre]:rounded-lg [&_pre]:whitespace-pre-wrap [&_pre]:break-words [&_code]:text-th-text-secondary [&_code]:break-words">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{linkifyContent(message.content)}</ReactMarkdown>
            </div>
          </div>
        </div>
      );
    }

    if (isThought) {
      return (
        <div className="ml-10">
          <div className="py-1">
            <div className="text-xs text-th-text-tertiary prose prose-sm max-w-none leading-relaxed break-words [&_p]:mb-1 [&_p:last-child]:mb-0 [&_pre]:bg-th-code-bg [&_pre]:border [&_pre]:border-th-border [&_pre]:rounded-lg [&_pre]:whitespace-pre-wrap [&_pre]:break-words [&_code]:text-th-text-secondary [&_code]:break-words">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{linkifyContent(message.content)}</ReactMarkdown>
            </div>
          </div>
        </div>
      );
    }

    const memoryTopics = message.metadata?.memory_topics as string[] | undefined;
    const rawThought = message.metadata?.thought as string | undefined;
    // Suppress trivial boilerplate thoughts (e.g. "done — greeting acknowledged")
    // that the ReAct prompt template encourages models to emit on simple replies.
    const thought = rawThought && !/^done\b/i.test(rawThought.trim()) ? rawThought.trim() : undefined;
    const stats = message.metadata?.stats as MlxStats | undefined;
    return (
      <div className="flex flex-col gap-2">
        {thought && (
          <details className="ml-11 group">
            <summary className="cursor-pointer text-[10px] uppercase tracking-wider text-th-text-muted hover:text-th-text-tertiary inline-flex items-center gap-1 list-none select-none [&::-webkit-details-marker]:hidden">
              <ChevronRight size={11} className="transition-transform group-open:rotate-90" />
              Thinking
            </summary>
            <div className="mt-1 ml-4 pl-3 border-l-2 border-th-border text-xs text-th-text-tertiary italic whitespace-pre-wrap leading-relaxed">
              {thought}
            </div>
          </details>
        )}
        <div className="flex gap-3 justify-start items-start">
          <div className="w-7 h-7 rounded-full border border-th-border/70 bg-th-inset-bg flex items-center justify-center shrink-0 mt-1">
            <Bot size={13} className="text-th-text-muted" />
          </div>
          <div className="min-w-0 flex-1 py-1">
            {memoryTopics && memoryTopics.length > 0 && (
              <div className="flex items-center gap-1.5 mb-3 flex-wrap">
                <span className="text-[10px] text-blue-400/70 font-medium uppercase tracking-wider">Memory</span>
                {memoryTopics.map((t) => (
                  <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400/80 border border-blue-500/15">{t.replace(/\.md$/, "")}</span>
                ))}
              </div>
            )}
            <div className={`
              prose max-w-[680px] break-words
              [font-family:'Source_Serif_4_Variable',Georgia,serif]
              text-th-text-primary
              prose-headings:text-th-text-primary prose-headings:font-semibold prose-headings:tracking-tight
              prose-headings:[font-family:'Inter_Variable',ui-sans-serif,system-ui,sans-serif]
              prose-h1:text-2xl prose-h1:mt-10 prose-h1:mb-5 prose-h1:leading-snug
              prose-h2:text-[19px] prose-h2:mt-8 prose-h2:mb-4 prose-h2:leading-snug
              prose-h3:text-[17px] prose-h3:mt-7 prose-h3:mb-3 prose-h3:leading-snug
              prose-h4:text-[15px] prose-h4:mt-5 prose-h4:mb-2 prose-h4:font-semibold
              prose-p:text-[16px] prose-p:leading-[1.85] prose-p:text-th-text-primary prose-p:my-5
              prose-strong:text-th-text-primary prose-strong:font-semibold
              prose-em:text-th-text-secondary
              prose-li:text-[16px] prose-li:leading-[1.8] prose-li:text-th-text-primary
              prose-ul:my-5 prose-ol:my-5
              prose-ul:pl-6 prose-ol:pl-6
              prose-li:my-2
              prose-blockquote:border-l-[3px] prose-blockquote:border-th-border-strong
              prose-blockquote:text-th-text-secondary prose-blockquote:not-italic prose-blockquote:pl-5 prose-blockquote:my-6 prose-blockquote:py-1
              prose-code:text-th-text-secondary prose-code:bg-th-code-bg prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md
              prose-code:text-[13px] prose-code:[font-family:ui-monospace,SFMono-Regular,Menlo,monospace]
              prose-code:font-normal prose-code:before:content-none prose-code:after:content-none
              prose-pre:bg-th-code-bg prose-pre:border prose-pre:border-th-border prose-pre:rounded-xl prose-pre:my-6 prose-pre:p-5
              prose-pre:text-[13px] prose-pre:leading-relaxed
              prose-pre:[font-family:ui-monospace,SFMono-Regular,Menlo,monospace]
              prose-table:text-[14px] prose-table:border-collapse
              prose-th:text-th-text-secondary prose-th:font-semibold prose-th:text-left prose-th:py-3 prose-th:px-4 prose-th:border-b prose-th:border-th-border
              prose-th:[font-family:'Inter_Variable',ui-sans-serif,system-ui,sans-serif]
              prose-td:text-th-text-primary prose-td:py-3 prose-td:px-4 prose-td:border-b prose-td:border-th-border/50
              prose-hr:border-th-border prose-hr:my-8
              prose-a:text-blue-400 prose-a:no-underline hover:prose-a:underline
            `}>
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{linkifyContent(message.content)}</ReactMarkdown>
            </div>
            {stats && (
              <div className="mt-2">
                <StatsChip stats={stats} />
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (message.type === "tool_call" || message.type === "tool_result") {
    const toolName = message.content;
    const args = message.metadata?.args as Record<string, unknown> | undefined;
    const displayLabel = getToolLabel(toolName, args);
    const isDone = message.type === "tool_result";
    const isStopped = isDone && message.metadata?.stopped === true;
    const hasResult = isDone && message.metadata?.result != null;
    const images = (message.metadata?.images as Array<{ base64: string; mime_type: string }>) ?? [];
    const hasImages = images.length > 0;
    const subagentName = message.metadata?.subagent as string | undefined;
    const isExecute = toolName === "execute";
    // Live output lines accumulated while the command runs
    const liveOutput = (message.metadata?.liveOutput as string[] | undefined) ?? [];
    const hasLiveOutput = !isDone && isExecute && liveOutput.length > 0;
    // Head/tail metadata for completed execute results
    const outputLines = message.metadata?.output_lines as number | undefined;
    const outputTruncated = message.metadata?.output_truncated as boolean | undefined;

    const isTodoTool = toolName === "write_todos";
    const todos = isTodoTool ? parseTodos(args) : null;

    const rawArtifactPath =
      isDone && !isStopped && toolName === "write_file"
        ? ((args?.path ?? args?.file_path) as string | undefined)
        : undefined;
    const artifactPath = typeof rawArtifactPath === "string" ? rawArtifactPath : null;
    const artifactType: ArtifactType | null = artifactPath ? artifactTypeFromPath(artifactPath) : null;
    const artifactFileUrl =
      artifactType && sessionId
        ? api.getSessionFileUrl(sessionId, artifactPath!.replace(/^\//, ""))
        : null;

    return (
      <div className={subagentName ? (inGroup ? "" : "ml-16 pl-3") : "ml-10"}>
        <button onClick={() => setExpanded(!expanded)} className="flex items-center gap-2 text-xs text-th-text-tertiary hover:text-th-text-secondary transition-colors py-1.5 group/tool">
          {expanded ? <ChevronUp size={12} /> : <ChevronRight size={12} />}
          {hasImages ? <Image size={12} className="text-blue-400" /> : (() => { const { Icon, className } = getToolIcon(toolName ?? ""); return <Icon size={12} className={className} />; })()}
          {subagentName && !inGroup && <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium shrink-0">{subagentName}</span>}
          <span className="text-th-text-tertiary group-hover/tool:text-th-text-secondary truncate max-w-[400px]">{displayLabel}</span>
          {!isDone && <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium animate-pulse shrink-0">running</span>}
          {isStopped && <span className="text-[10px] px-2 py-0.5 rounded-full bg-neutral-500/10 text-th-text-muted border border-neutral-500/20 font-medium shrink-0">stopped</span>}
          {isDone && !isStopped && <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium shrink-0">done</span>}
        </button>
        {todos && todos.length > 0 && isLatestTodo && (
          <TodoChecklist todos={todos} />
        )}
        {/* Live output tail — shown while the execute command is running */}
        {hasLiveOutput && (
          <div className="mt-1.5 ml-6">
            <div className="flex items-center gap-1.5 mb-1">
              <ScrollText size={10} className="text-th-text-muted shrink-0" />
              <span className="text-[10px] uppercase tracking-wider text-th-text-muted font-semibold">Live output</span>
              <span className="text-[10px] text-th-text-muted ml-auto">{liveOutput.length} lines</span>
            </div>
            <pre className="text-[11px] leading-[1.5] text-emerald-400/80 bg-[#0d1117] border border-th-border rounded-lg p-2.5 max-h-36 overflow-y-auto font-mono whitespace-pre-wrap break-all">
              {liveOutput.slice(-25).join("\n")}
            </pre>
          </div>
        )}
        {hasImages && (
          <div className="mt-1.5 ml-6 flex flex-wrap gap-2">
            {images.map((img, i) => {
              const src = `data:${img.mime_type};base64,${img.base64}`;
              const subtype = img.mime_type.split("/")[1] ?? "png";
              const ext = subtype === "jpeg" ? "jpg" : subtype === "svg+xml" ? "svg" : subtype;
              return (
                <button
                  key={i}
                  onClick={() => onOpenArtifact?.(`screenshot-${i + 1}.${ext}`, src, "image")}
                  className="relative w-[90px] h-[120px] rounded-lg border border-th-border overflow-hidden bg-th-inset-bg hover:border-blue-400/60 hover:scale-[1.02] transition-all focus:outline-none focus:ring-2 focus:ring-blue-400/50 shrink-0"
                  title={`View image ${i + 1}`}
                >
                  <img
                    src={src}
                    alt={`Tool result image ${i + 1}`}
                    className="w-full h-full object-cover"
                  />
                </button>
              );
            })}
          </div>
        )}
        {artifactType && artifactFileUrl && (
          <button
            onClick={() => onOpenArtifact?.(artifactPath!, artifactFileUrl, artifactType)}
            className="mt-2 ml-6 flex items-center gap-2.5 px-3 py-2 rounded-xl border border-th-border bg-th-card-bg hover:bg-th-surface-hover transition-colors group/artifact text-left"
          >
            {artifactType === "html"
              ? <Globe size={14} className="text-blue-400 shrink-0" />
              : artifactType === "pdf"
              ? <FileText size={14} className="text-red-400 shrink-0" />
              : artifactType === "image"
              ? <Image size={14} className="text-purple-400 shrink-0" />
              : <FileText size={14} className="text-emerald-400 shrink-0" />}
            <span className="text-xs font-medium text-th-text-primary truncate max-w-[240px]">
              {artifactPath!.split("/").pop()}
            </span>
            <span className="ml-auto text-[10px] text-th-text-muted group-hover/artifact:text-blue-400 transition-colors shrink-0">
              Open ↗
            </span>
          </button>
        )}
        {expanded && (
          <div className="mt-1.5 ml-6 space-y-2">
            {message.metadata?.args != null && (
              <div>
                <span className="text-[10px] uppercase tracking-wider text-th-text-muted font-semibold">Args</span>
                <pre className="mt-1 text-xs text-th-text-secondary bg-th-code-bg border border-th-border rounded-lg p-3 overflow-x-auto max-h-40 overflow-y-auto">
                  {JSON.stringify(message.metadata.args, null, 2)}
                </pre>
              </div>
            )}
            {hasResult && isExecute ? (
              <ExecuteResultBlock
                result={String(message.metadata!.result as string)}
                outputLines={outputLines}
                outputTruncated={outputTruncated}
              />
            ) : hasResult ? (
              <div>
                <span className="text-[10px] uppercase tracking-wider text-th-text-muted font-semibold">Result</span>
                <pre className="mt-1 text-xs text-th-text-secondary bg-th-code-bg border border-th-border rounded-lg p-3 overflow-x-auto max-h-40 overflow-y-auto">
                  {String(message.metadata!.result as string)}
                </pre>
              </div>
            ) : null}
          </div>
        )}
      </div>
    );
  }

  if (message.type === "ask_user") {
    const options = message.metadata?.options as string[] | undefined;
    const allowMultiple = message.metadata?.allow_multiple as boolean | undefined;
    const resolved = message.metadata?.resolved as boolean | undefined;
    const decisions = message.metadata?.decisions as Array<Record<string, unknown>> | undefined;
    const resolvedAnswer = decisions?.[0]?.answers
      ? (decisions[0].answers as string[]).join(", ")
      : (decisions?.[0]?.answer as string | undefined);

    return (
      <AskUserBubble
        question={message.content}
        options={options}
        allowMultiple={allowMultiple}
        resolved={resolved}
        resolvedAnswer={resolvedAnswer}
        onAnswer={(answer) => onHitlDecision?.([{ type: "ask_user_answer", answer }])}
        onAnswerMultiple={(answers) => onHitlDecision?.([{ type: "ask_user_answer", answers }])}
      />
    );
  }

  if (message.type === "hitl_request") {
    const meta = message.metadata as Record<string, unknown> | undefined;

    // Credential request — agent paused inside ``request_credential``.
    // Render a masked input dialog that writes the secret straight to
    // the OS keychain (``/api/vault/secrets/...``) before resuming the
    // graph, so the value never travels through the agent's message
    // history.
    if (meta?.type === "request_credential") {
      const decisions = meta.decisions as Array<Record<string, unknown>> | undefined;
      const resolved = meta.resolved as boolean | undefined;
      const resolvedAnswer = decisions?.[0]?.answer as string | undefined;

      return (
        <CredentialInputBubble
          serverId={String(meta.server_id ?? "")}
          name={String(meta.name ?? "")}
          displayLabel={String(meta.display_label ?? "Credential")}
          instructions={String(meta.instructions ?? "")}
          signupUrl={String(meta.signup_url ?? "")}
          resolved={resolved}
          resolvedAnswer={resolvedAnswer}
          onSubmit={(answer) => onHitlDecision?.([{ type: "credential_provided", answer }])}
        />
      );
    }

    const actions = (meta?.action_requests as Array<{ name: string; args: Record<string, unknown>; description?: string }>) || [];
    const resolved = meta?.resolved as boolean | undefined;
    const decisions = meta?.decisions as Array<Record<string, unknown>> | undefined;
    const resolvedType = decisions?.[0]?.type as string | undefined;

    return (
      <div className="flex gap-3 justify-start">
        <div className="max-w-[75%] w-full rounded-xl border border-th-border overflow-hidden">
          {/* Card header */}
          <div className="flex items-center gap-2 px-3 py-2 bg-th-inset-bg border-b border-th-border/60">
            {resolved
              ? resolvedType === "approve" || resolvedType === "edit"
                ? <ShieldCheck size={13} className="text-emerald-400 shrink-0" />
                : <ShieldX size={13} className="text-red-400 shrink-0" />
              : <ShieldAlert size={13} className="text-th-text-tertiary shrink-0" />}
            <span className={`text-xs font-medium flex-1 ${
              resolved
                ? resolvedType === "approve" || resolvedType === "edit"
                  ? "text-emerald-400/80"
                  : "text-red-400/80"
                : "text-th-text-secondary"
            }`}>
              {resolved
                ? resolvedType === "approve" ? "Approved" : resolvedType === "edit" ? "Approved (edited)" : "Rejected"
                : actions.length === 1 ? "Run command?" : `Run ${actions.length} commands?`}
            </span>
          </div>

          {/* Command blocks */}
          <div className="divide-y divide-th-border/40">
            {actions.map((action, i) => {
              const riskLabels = action.name === "execute"
                ? screenHighRiskCommand(action.args?.command)
                : [];
              const editedDecision = resolved && (decisions?.[i] as Record<string, any> | undefined)?.type === "edit" && decisions?.[i];
              const editedCmd = editedDecision ? (editedDecision as Record<string, any>).edited_action?.args?.command : undefined;
              const displayCmd = editedCmd ?? action.args?.command ?? JSON.stringify(action.args, null, 2);
              const isExecute = action.name === "execute";
              return (
                <div key={i} className="px-3 pt-2.5 pb-2.5">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <Terminal size={11} className="text-th-text-muted shrink-0" />
                    <span className="text-[11px] text-th-text-muted font-medium">{getToolLabel(action.name)}</span>
                    {riskLabels.length > 0 && (
                      <span
                        className="inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded-md bg-red-500/10 text-red-400 border border-red-500/20"
                        title={riskLabels.map(describeHighRiskLabel).join(" \u2022 ")}
                      >
                        <AlertTriangle size={9} />
                        High risk
                      </span>
                    )}
                    {editedCmd && (
                      <span className="text-[10px] text-blue-400/60 font-medium">edited</span>
                    )}
                  </div>

                  {hitlEditing && !resolved && i === 0 ? (
                    <textarea
                      className="w-full bg-th-input-bg border border-th-border rounded-lg px-3 py-2 text-sm font-mono text-th-text-primary focus:outline-none focus:border-blue-500/40 resize-none min-h-[60px]"
                      value={hitlEditCommand}
                      onChange={(e) => setHitlEditCommand(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Escape") setHitlEditing(false); }}
                      autoFocus
                    />
                  ) : (
                    <pre className={`text-sm font-mono text-th-text-primary bg-th-code-bg border rounded-lg px-3 py-2 overflow-x-auto whitespace-pre-wrap ${editedCmd ? "border-blue-500/30" : "border-th-border"}`}>
                      {isExecute && <span className="text-th-text-muted select-none">$ </span>}
                      {String(displayCmd)}
                    </pre>
                  )}

                  {riskLabels.length > 0 && !resolved && (
                    <p className="mt-1.5 text-[11px] text-red-400/70 flex items-start gap-1 leading-snug">
                      <AlertTriangle size={10} className="mt-0.5 shrink-0" />
                      {riskLabels.map(describeHighRiskLabel).join("; ")}. Review carefully before approving.
                    </p>
                  )}
                </div>
              );
            })}
          </div>

          {/* Action bar */}
          {!resolved && (
            <div className="flex items-center gap-1 px-3 py-2 border-t border-th-border/60 bg-th-inset-bg/50">
              {hitlEditing ? (
                <>
                  <button
                    onClick={() => {
                      if (hitlEditCommand.trim() && actions[0]) {
                        onHitlDecision?.(actions.map((a, idx) => idx === 0
                          ? { type: "edit", edited_action: { name: a.name, args: { ...a.args, command: hitlEditCommand.trim() } } }
                          : { type: "approve" }));
                        setHitlEditing(false);
                      }
                    }}
                    disabled={!hitlEditCommand.trim()}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 hover:bg-emerald-500/25 transition-colors disabled:opacity-40"
                  >
                    <Check size={11} /> Run edited
                  </button>
                  <button
                    onClick={() => setHitlEditing(false)}
                    className="px-3 py-1.5 text-xs rounded-md text-th-text-tertiary hover:text-th-text-secondary transition-colors"
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => onHitlDecision?.(actions.map(() => ({ type: "approve" })))}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-md bg-th-surface-hover text-th-text-primary border border-th-border hover:bg-th-surface-hover/70 transition-colors"
                  >
                    <Check size={11} /> {actions.length > 1 ? `Run (${actions.length})` : "Run"}
                  </button>
                  {actions.length === 1 && (
                    <button
                      onClick={() => {
                        const cmd = String(actions[0]?.args?.command ?? "");
                        setHitlEditCommand(cmd);
                        setHitlEditing(true);
                      }}
                      className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md text-th-text-tertiary hover:text-th-text-secondary hover:bg-th-surface-hover/50 transition-colors"
                    >
                      <Pencil size={11} /> Edit
                    </button>
                  )}
                  <button
                    onClick={() => onHitlDecision?.(actions.map(() => ({ type: "reject", message: "User rejected the command" })))}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md text-th-text-tertiary hover:text-red-400 hover:bg-red-500/5 transition-colors"
                  >
                    <X size={11} /> {actions.length > 1 ? "Reject all" : "Reject"}
                  </button>
                  <button
                    onClick={() => onApproveAllSession?.(actions.map(() => ({ type: "approve" })))}
                    title="Auto-approve all further commands this session"
                    className="ml-auto flex items-center gap-1 px-2 py-1.5 text-[11px] text-th-text-muted hover:text-th-text-tertiary transition-colors"
                  >
                    <CheckCheck size={10} /> Always allow
                  </button>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

  if (message.type === "error") {
    const isPrivacyLock = message.metadata?.error_code === "privacy_lock";
    if (isPrivacyLock) {
      const provider = message.metadata?.llm_provider as string | undefined;
      return (
        <div className="flex gap-3 justify-start">
          <div className="max-w-[70%] w-full rounded-xl border border-th-border overflow-hidden">
            <div className="flex items-center gap-2 px-3 py-2 bg-th-inset-bg border-b border-th-border/60">
              <ShieldAlert size={13} className="text-th-text-tertiary shrink-0" />
              <span className="text-xs font-medium text-th-text-secondary flex-1">Privacy Lock active</span>
              {provider && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-th-surface-hover/40 text-th-text-muted border border-th-border font-mono">
                  {provider}
                </span>
              )}
            </div>
            <div className="px-3 py-2.5">
              <p className="text-sm text-th-text-primary leading-relaxed mb-2">
                This session uses <strong>{provider ?? "a cloud provider"}</strong> which sends
                data off-device. The privacy lock is blocking the request.
              </p>
              <p className="text-xs text-th-text-tertiary leading-relaxed mb-3">
                Start a new session with a local provider (afm, mlx, omlx, exo),
                or disengage the lock to continue using cloud LLMs.
              </p>
              <Link
                to="/settings?tab=Privacy+%26+Security"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-th-surface-hover text-th-text-secondary border border-th-border hover:text-th-text-primary transition-colors"
              >
                <ExternalLink size={11} />
                Settings → Privacy &amp; Security
              </Link>
            </div>
          </div>
        </div>
      );
    }
    return <div className="bg-red-500/10 border border-red-500/25 rounded-xl px-4 py-3 text-sm text-red-400">{message.content}</div>;
  }

  // context_received — persisted echo of user-injected context; render as a
  // confirmed context bubble when loaded from history.
  if (message.type === "context_received") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] min-w-0">
          <div className="bg-violet-500/5 border border-violet-500/20 rounded-2xl rounded-br-sm px-4 py-3">
            <div className="flex items-center gap-1.5 mb-1.5">
              <MessageSquarePlus size={11} className="text-violet-400/70 shrink-0" />
              <span className="text-[10px] text-violet-400/70 font-medium uppercase tracking-wide">Context added</span>
            </div>
            <div className="text-sm text-th-text-primary whitespace-pre-wrap break-words leading-relaxed">
              {message.content}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
});


// ---------------------------------------------------------------------------
// StatsChip — compact MLX generation stats pill rendered below the bubble
// ---------------------------------------------------------------------------

function StatsChip({ stats }: { stats: MlxStats }) {
  const cached = stats.tokens_from_cache ?? 0;
  const prefilled = stats.tokens_prefilled ?? 0;
  const total = cached + prefilled;
  const hitPct = stats.cache_hit_ratio != null ? Math.round(stats.cache_hit_ratio * 100) : null;

  // Compact summary shown as the pill label.  Falls back to whatever data
  // is available so the chip stays useful even if cache/prefill stats are
  // missing (e.g. first-turn cold cache or a non-cached model).
  const compactParts: string[] = [];
  if (hitPct != null && total > 0) compactParts.push(`cache ${hitPct}%`);
  if (stats.prompt_tps) compactParts.push(`prefill ${stats.prompt_tps} t/s`);
  if (stats.generation_tps) compactParts.push(`gen ${stats.generation_tps} t/s`);
  if (stats.peak_memory_gb) compactParts.push(`${stats.peak_memory_gb.toFixed(1)} GB`);
  if (compactParts.length === 0) return null;

  // Full breakdown shown in the native tooltip on hover.
  const tooltipLines: string[] = [];
  if (total > 0 && hitPct != null)
    tooltipLines.push(`Cache hit: ${cached}/${total} tokens (${hitPct}%)`);
  if (stats.prompt_tps) tooltipLines.push(`Prefill: ${stats.prompt_tps} t/s`);
  if (stats.generation_tps && stats.generation_tokens)
    tooltipLines.push(`Generation: ${stats.generation_tokens} tok @ ${stats.generation_tps} t/s`);
  if (stats.peak_memory_gb) tooltipLines.push(`Peak GPU memory: ${stats.peak_memory_gb.toFixed(2)} GB`);

  return (
    <div
      className="mt-1 inline-flex items-center gap-1 px-1.5 py-px rounded-full bg-th-surface-hover/40 border border-th-border text-[8px] text-th-text-primary font-mono self-start cursor-default hover:bg-th-surface-hover/70 transition-colors"
      title={tooltipLines.join("\n")}
    >
      {compactParts.join(" · ")}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ExecuteResultBlock — renders execute tool results with head/tail display
// ---------------------------------------------------------------------------

const OMIT_MARKER_RE = /\n⋯ \((\d+) more lines\) ⋯\n/;

function ExecuteResultBlock({
  result,
  outputLines,
  outputTruncated,
}: {
  result: string;
  outputLines?: number;
  outputTruncated?: boolean;
}) {
  const [showFull, setShowFull] = useState(false);

  if (!outputTruncated || showFull) {
    return (
      <div>
        <div className="flex items-center gap-1.5 mb-1">
          <span className="text-[10px] uppercase tracking-wider text-th-text-muted font-semibold">Result</span>
          {outputLines != null && (
            <span className="text-[10px] text-th-text-muted ml-auto">{outputLines} lines</span>
          )}
        </div>
        <pre className="text-xs text-th-text-secondary bg-th-code-bg border border-th-border rounded-lg p-3 overflow-x-auto max-h-72 overflow-y-auto whitespace-pre-wrap break-all">
          {result}
        </pre>
      </div>
    );
  }

  const match = OMIT_MARKER_RE.exec(result);
  if (!match) {
    return (
      <div>
        <span className="text-[10px] uppercase tracking-wider text-th-text-muted font-semibold">Result</span>
        <pre className="mt-1 text-xs text-th-text-secondary bg-th-code-bg border border-th-border rounded-lg p-3 overflow-x-auto max-h-40 overflow-y-auto whitespace-pre-wrap break-all">
          {result}
        </pre>
      </div>
    );
  }

  const omittedCount = Number(match[1]);
  const splitIdx = match.index;
  const head = result.slice(0, splitIdx);
  const tail = result.slice(splitIdx + match[0].length);

  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-[10px] uppercase tracking-wider text-th-text-muted font-semibold">Result</span>
        {outputLines != null && (
          <span className="text-[10px] text-th-text-muted ml-auto">{outputLines} lines</span>
        )}
      </div>
      <div className="rounded-lg border border-th-border bg-th-code-bg overflow-hidden">
        <pre className="text-xs text-th-text-secondary p-3 overflow-x-auto whitespace-pre-wrap break-all max-h-40 overflow-y-auto">
          {head}
        </pre>
        <button
          onClick={() => setShowFull(true)}
          className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 border-y border-th-border bg-th-surface-hover/40 hover:bg-th-surface-hover text-[10px] text-th-text-muted hover:text-th-text-secondary transition-colors"
        >
          <ChevronDown size={10} />
          {omittedCount} lines omitted — click to expand
        </button>
        <pre className="text-xs text-th-text-secondary p-3 overflow-x-auto whitespace-pre-wrap break-all max-h-40 overflow-y-auto border-t border-th-border">
          {tail}
        </pre>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// TodoChecklist — renders write_todos as a Cursor-style task list
// ---------------------------------------------------------------------------

interface TodoItem {
  content: string;
  status: "pending" | "in_progress" | "completed" | "cancelled";
}

function parseTodos(args: Record<string, unknown> | undefined): TodoItem[] | null {
  if (!args) return null;
  const raw = args.todos;
  if (!Array.isArray(raw)) return null;
  return raw
    .filter((t): t is Record<string, unknown> => t != null && typeof t === "object")
    .map((t) => ({
      content: String(t.content ?? ""),
      status: (["pending", "in_progress", "completed", "cancelled"].includes(String(t.status))
        ? String(t.status)
        : "pending") as TodoItem["status"],
    }));
}

const STATUS_CONFIG = {
  completed: {
    icon: CheckCircle2,
    iconClass: "text-emerald-400",
    textClass: "text-th-text-tertiary line-through",
    bgClass: "bg-emerald-500/5",
  },
  in_progress: {
    icon: Loader2,
    iconClass: "text-blue-400 animate-spin",
    textClass: "text-th-text-primary",
    bgClass: "bg-blue-500/5",
  },
  pending: {
    icon: Circle,
    iconClass: "text-th-text-muted",
    textClass: "text-th-text-tertiary",
    bgClass: "",
  },
  cancelled: {
    icon: XCircle,
    iconClass: "text-th-text-muted",
    textClass: "text-th-text-muted line-through",
    bgClass: "",
  },
} as const;

function TodoChecklist({ todos }: { todos: TodoItem[] }) {
  const done = todos.filter((t) => t.status === "completed").length;
  return (
    <div className="mt-1.5 ml-6 rounded-lg border border-th-border bg-th-card-bg/90 overflow-hidden max-w-md">
      <div className="flex items-center gap-2.5 px-3 py-2 border-b border-th-border bg-th-inset-bg">
        <span className="text-[11px] font-semibold text-th-text-tertiary tracking-wide">Tasks</span>
        <span className="text-[10px] text-th-text-muted font-medium ml-auto">{done}/{todos.length} completed</span>
        <div className="w-16 h-1 rounded-full bg-th-border overflow-hidden">
          <div
            className="h-full rounded-full bg-emerald-500 transition-all duration-300"
            style={{ width: `${todos.length ? (done / todos.length) * 100 : 0}%` }}
          />
        </div>
      </div>
      <ul className="divide-y divide-th-border">
        {todos.map((todo, i) => {
          const cfg = STATUS_CONFIG[todo.status];
          const Icon = cfg.icon;
          return (
            <li
              key={i}
              className={`flex items-start gap-2.5 px-3 py-2 ${cfg.bgClass}`}
            >
              <Icon size={14} className={`shrink-0 mt-0.5 ${cfg.iconClass}`} />
              <span className={`text-xs leading-relaxed ${cfg.textClass}`}>{todo.content}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

// ---------------------------------------------------------------------------
// AskUserBubble — renders open-ended or multiple-choice questions
// ---------------------------------------------------------------------------

interface AskUserBubbleProps {
  question: string;
  options?: string[];
  allowMultiple?: boolean;
  resolved?: boolean;
  resolvedAnswer?: string;
  onAnswer: (answer: string) => void;
  onAnswerMultiple: (answers: string[]) => void;
}

const OTHER_LABEL = "Other…";

function AskUserBubble({
  question,
  options: rawOptions,
  allowMultiple,
  resolved,
  resolvedAnswer,
  onAnswer,
  onAnswerMultiple,
}: AskUserBubbleProps) {
  const [freeText, setFreeText] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  // When the user picks "Other…" in single-select mode, show an inline input
  const [showOtherInput, setShowOtherInput] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Always append "Other…" if options are provided and it's not already there
  const options = rawOptions
    ? rawOptions.includes(OTHER_LABEL)
      ? rawOptions
      : [...rawOptions, OTHER_LABEL]
    : undefined;

  const toggleOption = (opt: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(opt)) next.delete(opt);
      else next.add(opt);
      return next;
    });
  };

  // Focus the other-input when it appears
  useEffect(() => {
    if (showOtherInput) inputRef.current?.focus();
  }, [showOtherInput]);

  const submitOtherText = () => {
    const val = freeText.trim();
    if (val) onAnswer(val);
  };

  return (
    <div className="flex gap-3 justify-start">
      <div className="max-w-[70%] w-full rounded-xl border border-th-border overflow-hidden">
        {/* Header */}
        <div className="flex items-center gap-2 px-3 py-2 bg-th-inset-bg border-b border-th-border/60">
          {resolved
            ? <CheckCircle2 size={13} className="text-emerald-400 shrink-0" />
            : <MessageCircleQuestion size={13} className="text-th-text-tertiary shrink-0" />}
          <span className={`text-xs font-medium ${resolved ? "text-emerald-400/80" : "text-th-text-secondary"}`}>
            {resolved ? "Answered" : "Question"}
          </span>
        </div>

        {/* Body */}
        <div className="px-3 py-2.5">
          <p className="text-sm text-th-text-primary leading-relaxed mb-2.5">{question}</p>

          {resolved ? (
            <div className="px-3 py-2 bg-th-inset-bg border border-th-border rounded-lg">
              <span className="text-xs text-th-text-secondary">{resolvedAnswer}</span>
            </div>
          ) : options ? (
            <>
              <div className="flex flex-wrap gap-1.5">
                {options.map((opt) => {
                  const isOther = opt === OTHER_LABEL;
                  const isSelected = selected.has(opt);
                  return allowMultiple ? (
                    <button
                      key={opt}
                      onClick={() => toggleOption(opt)}
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md border transition-colors ${
                        isSelected
                          ? "bg-th-surface-hover text-th-text-primary border-th-border"
                          : "bg-th-inset-bg text-th-text-secondary border-th-border hover:border-th-border hover:text-th-text-primary hover:bg-th-surface-hover/50"
                      }`}
                    >
                      {isSelected && <Check size={10} />}
                      {opt}
                    </button>
                  ) : (
                    <button
                      key={opt}
                      onClick={() => isOther ? setShowOtherInput(true) : onAnswer(opt)}
                      className="px-3 py-1.5 text-xs font-medium rounded-md bg-th-inset-bg text-th-text-secondary border border-th-border hover:text-th-text-primary hover:bg-th-surface-hover/50 transition-colors"
                    >
                      {opt}
                    </button>
                  );
                })}
              </div>

              {showOtherInput && !allowMultiple && (
                <form
                  onSubmit={(e) => { e.preventDefault(); submitOtherText(); }}
                  className="flex gap-2 mt-2"
                >
                  <textarea
                    ref={inputRef}
                    className="flex-1 bg-th-input-bg border border-th-border rounded-lg px-3 py-2 text-sm text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-500/40 resize-none min-h-[40px] max-h-[120px]"
                    value={freeText}
                    onChange={(e) => setFreeText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        submitOtherText();
                      }
                    }}
                    onInput={(e) => {
                      const t = e.currentTarget;
                      t.style.height = "auto";
                      t.style.height = `${t.scrollHeight}px`;
                    }}
                    placeholder="Type your answer…"
                  />
                  <button
                    type="submit"
                    disabled={!freeText.trim()}
                    className="self-end w-8 h-8 rounded-lg flex items-center justify-center bg-th-inset-bg border border-th-border hover:bg-th-surface-hover transition-colors disabled:opacity-40"
                  >
                    <Send size={13} className="text-th-text-secondary" />
                  </button>
                </form>
              )}

              {allowMultiple && (
                <button
                  onClick={() => { if (selected.size > 0) onAnswerMultiple(Array.from(selected)); }}
                  disabled={selected.size === 0}
                  className="flex items-center gap-1.5 mt-2 px-3 py-1.5 text-xs font-medium rounded-md bg-th-surface-hover text-th-text-primary border border-th-border hover:bg-th-surface-hover/70 transition-colors disabled:opacity-40"
                >
                  <Check size={11} /> Submit{selected.size > 0 ? ` (${selected.size})` : ""}
                </button>
              )}
            </>
          ) : (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                const val = freeText.trim();
                if (val) onAnswer(val);
              }}
              className="flex gap-2"
            >
              <textarea
                ref={inputRef}
                className="flex-1 bg-th-input-bg border border-th-border rounded-lg px-3 py-2 text-sm text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-500/40 resize-none min-h-[40px] max-h-[120px]"
                value={freeText}
                onChange={(e) => setFreeText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    const val = freeText.trim();
                    if (val) onAnswer(val);
                  }
                }}
                onInput={(e) => {
                  const t = e.currentTarget;
                  t.style.height = "auto";
                  t.style.height = `${t.scrollHeight}px`;
                }}
                placeholder="Type your answer..."
                autoFocus
              />
              <button
                type="submit"
                disabled={!freeText.trim()}
                className="self-end w-8 h-8 rounded-lg flex items-center justify-center bg-th-inset-bg border border-th-border hover:bg-th-surface-hover transition-colors disabled:opacity-40"
              >
                <Send size={13} className="text-th-text-secondary" />
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CredentialInputBubble — masked input dialog for ``request_credential``.
// Writes the secret directly to the OS keychain via the vault API and only
// then resumes the LangGraph interrupt with a non-sensitive ack ("stored"
// or "cancelled").  The plaintext value is never forwarded to the agent;
// the tool re-reads the vault as the source of truth.
// ---------------------------------------------------------------------------

interface CredentialInputBubbleProps {
  serverId: string;
  name: string;
  displayLabel: string;
  instructions: string;
  signupUrl: string;
  resolved?: boolean;
  resolvedAnswer?: string;
  onSubmit: (answer: "stored" | "cancelled") => void;
}

function CredentialInputBubble({
  serverId,
  name,
  displayLabel,
  instructions,
  signupUrl,
  resolved,
  resolvedAnswer,
  onSubmit,
}: CredentialInputBubbleProps) {
  const [value, setValue] = useState("");
  const [reveal, setReveal] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = async () => {
    const trimmed = value.trim();
    if (!trimmed || saving || !serverId || !name) return;
    setSaving(true);
    setError(null);
    try {
      await api.vaultSetSecret(serverId, name, trimmed);
      onSubmit("stored");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSaving(false);
    }
  };

  const handleCancel = () => {
    if (saving) return;
    onSubmit("cancelled");
  };

  const headerIcon = resolved
    ? resolvedAnswer === "stored"
      ? <ShieldCheck size={13} className="text-emerald-400 shrink-0" />
      : <ShieldX size={13} className="text-red-400 shrink-0" />
    : <KeyRound size={13} className="text-th-text-tertiary shrink-0" />;

  const headerLabel = resolved
    ? resolvedAnswer === "stored" ? "Saved to keychain" : "Cancelled"
    : "Credential required";

  const headerTone = resolved
    ? resolvedAnswer === "stored" ? "text-emerald-400/80" : "text-red-400/80"
    : "text-th-text-secondary";

  return (
    <div className="flex gap-3 justify-start">
      <div className="max-w-[70%] w-full rounded-xl border border-th-border overflow-hidden">
        {/* Header */}
        <div className="flex items-center gap-2 px-3 py-2 bg-th-inset-bg border-b border-th-border/60">
          {headerIcon}
          <span className={`text-xs font-medium flex-1 ${headerTone}`}>{headerLabel}</span>
          {serverId && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-th-surface-hover/40 text-th-text-muted border border-th-border font-mono">
              {serverId}
            </span>
          )}
        </div>

        {/* Body */}
        <div className="px-3 py-2.5">
          <p className="text-sm text-th-text-primary mb-0.5 leading-relaxed">
            {displayLabel}
            {name && (
              <span className="ml-2 text-[10px] font-mono text-th-text-muted">{name}</span>
            )}
          </p>
          {instructions && (
            <p className="text-xs text-th-text-tertiary mb-2.5 leading-relaxed">{instructions}</p>
          )}

          {resolved ? (
            <div className="px-3 py-2 bg-th-inset-bg border border-th-border rounded-lg">
              <span className="text-xs text-th-text-secondary">
                {resolvedAnswer === "stored"
                  ? "Secret saved to your OS keychain. The agent only sees a 'stored' ack."
                  : "User cancelled — no value was saved."}
              </span>
            </div>
          ) : (
            <>
              <div className="flex gap-2 mt-2">
                <input
                  type={reveal ? "text" : "password"}
                  value={value}
                  onChange={(e) => setValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") { e.preventDefault(); void handleSave(); }
                    if (e.key === "Escape") { e.preventDefault(); handleCancel(); }
                  }}
                  placeholder={`Paste ${displayLabel.toLowerCase()}…`}
                  autoFocus
                  spellCheck={false}
                  autoComplete="off"
                  className="flex-1 bg-th-input-bg border border-th-border rounded-lg px-3 py-2 text-sm font-mono text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-500/40"
                />
                <button
                  type="button"
                  onClick={() => setReveal((r) => !r)}
                  title={reveal ? "Hide value" : "Show value"}
                  className="px-2 text-[10px] uppercase tracking-wider rounded-lg text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover border border-th-border transition-colors"
                >
                  {reveal ? "Hide" : "Show"}
                </button>
              </div>

              {error && (
                <div className="mt-2 text-xs text-red-400 flex items-start gap-1.5">
                  <AlertTriangle size={11} className="mt-0.5 shrink-0" />
                  <span>Failed to save: {error}</span>
                </div>
              )}

              <div className="flex items-center gap-2 mt-2.5">
                <button
                  onClick={() => void handleSave()}
                  disabled={!value.trim() || saving}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-th-surface-hover text-th-text-primary border border-th-border hover:bg-th-surface-hover/70 transition-colors disabled:opacity-40"
                >
                  {saving ? <Loader2 size={11} className="animate-spin" /> : <Check size={11} />}
                  {saving ? "Saving…" : "Save to keychain"}
                </button>
                <button
                  onClick={handleCancel}
                  disabled={saving}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md text-th-text-tertiary hover:text-th-text-secondary hover:bg-th-surface-hover/50 transition-colors disabled:opacity-40"
                >
                  <X size={11} /> Cancel
                </button>
                {signupUrl && (
                  <a
                    href={signupUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-auto flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md text-th-text-tertiary hover:text-th-text-secondary transition-colors"
                  >
                    <ExternalLink size={11} /> Sign up
                  </a>
                )}
              </div>

              <p className="mt-2 text-[10px] text-th-text-muted leading-relaxed">
                Stored locally in your OS keychain. The agent only sees a "stored" ack — never the value.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
