import { useState } from "react";
import { ChevronRight, ChevronDown, Clock, Bot } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { TimelineEvent } from "../../types";
import { getToolIcon } from "../../utils/entityIcons";

interface RunTimelineProps {
  events: TimelineEvent[];
}

function formatTs(ts?: string): string {
  if (!ts) return "";
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatDuration(ms?: number): string {
  if (ms == null) return "";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

type AugmentedEvent = TimelineEvent & { _result?: TimelineEvent };

// ── Rail node marker ─────────────────────────────────────────────────────────
// A small dot centered on the timeline spine, colored by event type so the
// sequence is scannable at a glance.
function NodeDot({ event }: { event: AugmentedEvent }) {
  const type = event.type;
  if (type === "user") {
    return <span className="block w-2.5 h-2.5 rounded-full bg-th-bg-secondary border-2 border-th-border-strong" />;
  }
  if (type === "assistant") {
    return <span className="block w-2.5 h-2.5 rounded-full bg-blue-400 shadow-[0_0_0_3px_rgba(59,130,246,0.16)]" />;
  }
  if (type === "tool_call" || type === "tool_result") {
    const pending = type === "tool_call" && !event._result;
    return <span className={`block w-2 h-2 rounded-full ${pending ? "bg-amber-400" : "bg-th-text-muted/50"}`} />;
  }
  if (type === "system" || type === "status") {
    return <span className="block w-1.5 h-1.5 rounded-full bg-th-border" />;
  }
  if (type === "done") return <span className="block w-3 h-3 rounded-full bg-emerald-400 shadow-[0_0_0_3px_rgba(52,211,153,0.16)]" />;
  if (type === "stopped") return <span className="block w-3 h-3 rounded-full bg-th-text-muted" />;
  if (type === "error") return <span className="block w-3 h-3 rounded-full bg-red-400 shadow-[0_0_0_3px_rgba(248,113,113,0.16)]" />;
  return <span className="block w-2 h-2 rounded-full bg-th-border" />;
}

// ── Per-event content (the right side of the rail) ────────────────────────────
function EventContent({ event, inGroup }: { event: AugmentedEvent; inGroup: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const type = event.type;
  const content = typeof event.content === "string" ? event.content : JSON.stringify(event.content);

  if (type === "user") {
    return (
      <>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[11px] font-semibold text-th-text-muted uppercase tracking-wider">User</span>
          {event.ts && <span className="text-[10px] text-th-text-muted/50 tabular-nums">{formatTs(event.ts)}</span>}
        </div>
        <p className="text-sm text-th-text-primary leading-relaxed whitespace-pre-wrap break-words">{content}</p>
      </>
    );
  }

  if (type === "assistant") {
    return (
      <>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[11px] font-semibold text-blue-400 uppercase tracking-wider">
            {event.subagent && !inGroup ? event.subagent : "Agent"}
          </span>
          {event.ts && <span className="text-[10px] text-th-text-muted/50 tabular-nums">{formatTs(event.ts)}</span>}
        </div>
        <div className="prose prose-sm max-w-none text-th-text-primary prose-p:my-1 prose-pre:text-xs prose-pre:bg-th-inset-bg">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      </>
    );
  }

  if (type === "tool_call" || type === "tool_result") {
    const toolName = event.tool || (typeof event.content === "string" ? event.content : "tool");
    const { Icon, className } = getToolIcon(toolName);
    const hasArgs = event.args && Object.keys(event.args).length > 0;
    const resultEvent = event._result;
    const resultContent = resultEvent
      ? (typeof resultEvent.content === "string" ? resultEvent.content : JSON.stringify(resultEvent.content))
      : (type === "tool_result" ? content : "");
    const hasResult = resultContent && resultContent.length > 0;
    const isPending = type === "tool_call" && !resultEvent;
    const durationMs = resultEvent?.duration_ms ?? event.duration_ms;
    const canExpand = hasArgs || hasResult;

    return (
      <>
        <button
          onClick={() => canExpand && setExpanded(!expanded)}
          className={`flex items-center gap-2 text-left w-full group/row rounded-lg -mx-2 px-2 py-1 transition-colors ${canExpand ? "hover:bg-th-surface-hover/50 cursor-pointer" : "cursor-default"}`}
        >
          <Icon size={12} className={`${className} shrink-0`} aria-hidden />
          {event.subagent && !inGroup && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium shrink-0">
              {event.subagent}
            </span>
          )}
          <span className="text-xs font-medium text-th-text-secondary group-hover/row:text-th-text-primary transition-colors truncate">
            {toolName}
          </span>
          {durationMs != null && (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-th-text-muted shrink-0 tabular-nums">
              <Clock size={9} aria-hidden />
              {formatDuration(durationMs)}
            </span>
          )}
          {isPending ? (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 font-medium shrink-0">running</span>
          ) : (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium shrink-0">done</span>
          )}
          {(resultEvent?.ts ?? event.ts) && (
            <span className="text-[10px] text-th-text-muted/50 ml-auto shrink-0 tabular-nums">{formatTs(resultEvent?.ts ?? event.ts)}</span>
          )}
          {canExpand && (
            expanded ? <ChevronDown size={11} className="text-th-text-muted shrink-0" /> : <ChevronRight size={11} className="text-th-text-muted shrink-0" />
          )}
        </button>
        {expanded && (
          <div className="mt-2 space-y-2">
            {hasArgs && (
              <div>
                <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-wider mb-1">Args</p>
                <pre className="text-[11px] bg-th-inset-bg border border-th-border/60 rounded-md p-2 overflow-x-auto text-th-text-secondary leading-relaxed">
                  {JSON.stringify(event.args, null, 2)}
                </pre>
              </div>
            )}
            {hasResult && (
              <div>
                <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-wider mb-1">Result</p>
                <p className="text-[11px] text-th-text-muted bg-th-inset-bg border border-th-border/60 rounded-md p-2 leading-relaxed whitespace-pre-wrap break-words">
                  {resultContent.slice(0, 800)}{resultContent.length > 800 ? "…" : ""}
                </p>
              </div>
            )}
          </div>
        )}
      </>
    );
  }

  if (type === "system" || type === "status") {
    return (
      <span className="text-[10px] text-th-text-muted/60 font-medium leading-relaxed">{content.slice(0, 120)}</span>
    );
  }

  if (type === "error") {
    return (
      <>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-semibold text-red-400">Error</span>
          {event.ts && <span className="text-[10px] text-th-text-muted/50 ml-auto tabular-nums">{formatTs(event.ts)}</span>}
        </div>
        <p className="text-[11px] text-red-300/80 bg-red-500/[0.07] border border-red-500/20 rounded-lg px-3 py-2 leading-relaxed break-words whitespace-pre-wrap">
          {content}
        </p>
      </>
    );
  }

  if (type === "done" || type === "stopped") {
    return (
      <div className="flex items-center gap-2">
        <span className={`text-xs font-medium ${type === "stopped" ? "text-th-text-muted" : "text-emerald-400"}`}>
          {type === "done" ? "Run completed" : "Run stopped"}
        </span>
        {event.ts && <span className="text-[10px] text-th-text-muted/50 ml-auto tabular-nums">{formatTs(event.ts)}</span>}
      </div>
    );
  }

  return null;
}

// ── A single row: rail gutter (dot) + content ────────────────────────────────
function TimelineRow({ event, inGroup = false }: { event: AugmentedEvent; inGroup?: boolean }) {
  const compact = event.type === "tool_call" || event.type === "tool_result" || event.type === "system" || event.type === "status";
  return (
    <div className="flex gap-3">
      <div className="relative z-10 flex w-[22px] shrink-0 justify-center">
        <span className="mt-[11px]">
          <NodeDot event={event} />
        </span>
      </div>
      <div className={`flex-1 min-w-0 pt-1 ${compact ? "pb-2" : "pb-3.5"}`}>
        <EventContent event={event} inGroup={inGroup} />
      </div>
    </div>
  );
}

// ── A delegated subagent's steps, branched off the main rail ──────────────────
function SubagentGroup({ name, events }: { name: string; events: AugmentedEvent[] }) {
  return (
    <div className="ml-7 my-1">
      <div className="flex items-center gap-2 mb-1.5">
        <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-md bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium">
          <Bot size={10} aria-hidden />
          {name}
        </span>
        <span className="text-[10px] text-th-text-muted/50 tabular-nums">{events.length} step{events.length !== 1 ? "s" : ""}</span>
      </div>
      <div className="relative">
        <div className="absolute left-[11px] top-2 bottom-2 w-px bg-blue-500/25" aria-hidden />
        {events.map((ev, i) => (
          <TimelineRow key={i} event={ev} inGroup />
        ))}
      </div>
    </div>
  );
}

function mergeToolEvents(events: TimelineEvent[]): AugmentedEvent[] {
  // Index all tool_result events by their tool_call_id for O(1) lookup.
  const resultById = new Map<string, TimelineEvent>();
  // Track tool_result events matched to a call so we can skip them in output.
  const consumedResultIndices = new Set<number>();

  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    if (ev.type === "tool_result" && ev.tool_call_id) {
      resultById.set(ev.tool_call_id, ev);
      consumedResultIndices.add(i);
    }
  }

  // For tool_result events that have no tool_call_id, pair them sequentially
  // with the nearest preceding unmatched tool_call.
  const unmatchedCallIndices: number[] = [];
  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    if (ev.type === "tool_call" && !resultById.has(ev.tool_call_id ?? "")) {
      unmatchedCallIndices.push(i);
    }
  }
  let callIdx = 0;
  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    if (ev.type === "tool_result" && !ev.tool_call_id && callIdx < unmatchedCallIndices.length) {
      resultById.set(`__seq_${unmatchedCallIndices[callIdx]}`, ev);
      consumedResultIndices.add(i);
      callIdx++;
    }
  }

  const merged: AugmentedEvent[] = [];
  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    if (consumedResultIndices.has(i)) continue; // skip — will appear inside its call row
    if (ev.type === "tool_call") {
      const result = ev.tool_call_id
        ? resultById.get(ev.tool_call_id)
        : resultById.get(`__seq_${i}`);
      merged.push(result ? { ...ev, _result: result } : ev);
    } else {
      merged.push(ev);
    }
  }
  return merged;
}

// Group consecutive same-subagent events so delegated work reads as one branch.
type RenderItem =
  | { kind: "event"; event: AugmentedEvent }
  | { kind: "group"; name: string; events: AugmentedEvent[] };

function groupEvents(merged: AugmentedEvent[]): RenderItem[] {
  const items: RenderItem[] = [];
  let current: { name: string; events: AugmentedEvent[] } | null = null;

  const flush = () => {
    if (current) {
      items.push({ kind: "group", name: current.name, events: current.events });
      current = null;
    }
  };

  for (const ev of merged) {
    if (ev.subagent) {
      if (current && current.name === ev.subagent) {
        current.events.push(ev);
      } else {
        flush();
        current = { name: ev.subagent, events: [ev] };
      }
    } else {
      flush();
      items.push({ kind: "event", event: ev });
    }
  }
  flush();
  return items;
}

export function RunTimeline({ events }: RunTimelineProps) {
  if (events.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <p className="text-sm text-th-text-muted">No timeline events found</p>
        <p className="text-xs text-th-text-muted/60 mt-1">Events are captured from the session transcript</p>
      </div>
    );
  }

  const merged = mergeToolEvents(events);
  const items = groupEvents(merged);

  return (
    <div className="relative">
      {/* Continuous timeline spine */}
      <div className="absolute left-[11px] top-3 bottom-3 w-px bg-th-border/50" aria-hidden />
      {items.map((item, i) =>
        item.kind === "group" ? (
          <SubagentGroup key={i} name={item.name} events={item.events} />
        ) : (
          <TimelineRow key={i} event={item.event} />
        ),
      )}
    </div>
  );
}
