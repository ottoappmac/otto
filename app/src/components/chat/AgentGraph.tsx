import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, GitBranch, X, ZoomIn, ZoomOut, Maximize2, RotateCcw } from "lucide-react";
import type { ChatMessage } from "../../types";
import { getAgentIcon, getToolIcon } from "../../utils/entityIcons";

// ── Layout constants ──────────────────────────────────────────────────────────

const NW  = 152; // subagent / orchestrator node width
const NH  = 48;  // subagent / orchestrator node height
const TW  = 138; // tool node width
const TH  = 48;  // tool node height
const HG  = 14;  // horizontal gap between nodes
const VG  = 56;  // vertical gap between ranks
const TVH = 6;   // gap between tool rows
const TVW = 6;   // gap between tools in a row
const MAX_TOOLS_PER_ROW = 5;
const MAX_SA_PER_ROW    = 5;
const SA_ROW_GAP        = 24;
const PAD = 18;
// Max chars kept for a tool node's result preview. Generous enough to show
// multi-line command (execute) output in the detail panel, which renders it
// in a scrollable block with line breaks preserved.
const RESULT_PREVIEW_MAX = 2000;

// ── Internal types ────────────────────────────────────────────────────────────

interface GNode {
  id: string;
  label: string;
  kind: "orchestrator" | "subagent" | "tool";
  status: "done" | "running";
  toolCount: number;
  args?: Record<string, unknown>;
  resultPreview?: string;
  uniqueTools?: string[];
  /** Execution order of this action (1-based) within the run. */
  seq?: number;
  /** Wall-clock duration in ms, when known (from the timeline). */
  durationMs?: number;
}

interface GEdge { from: string; to: string }

// ── Graph builder ─────────────────────────────────────────────────────────────

function buildGraph(
  messages: ChatMessage[],
  durations: Record<string, number> = {},
): { nodes: GNode[]; edges: GEdge[] } {
  const nodeMap = new Map<string, GNode>();
  const edges: GEdge[] = [];
  const edgeSet = new Set<string>();

  const addEdge = (f: string, t: string) => {
    const k = `${f}→${t}`;
    if (!edgeSet.has(k)) { edgeSet.add(k); edges.push({ from: f, to: t }); }
  };

  nodeMap.set("orch", {
    id: "orch", label: "Orchestrator", kind: "orchestrator", status: "running", toolCount: 0,
  });

  const taskCalls      = new Map<string, { type: string; nodeId: string }>();
  const displayToNode  = new Map<string, string>();
  const tcToTool       = new Map<string, string>();
  let autoId = 0;
  // Monotonic execution-order counter assigned to each action (delegation or
  // tool call) in the order it appears in the transcript.
  let seq = 0;

  const sessionDone = messages.some(m => m.type === "done");

  for (const msg of messages) {
    const meta       = (msg.metadata ?? {}) as Record<string, unknown>;
    const subDisplay = meta.subagent as string | undefined;
    const tcId       = meta.tool_call_id as string | undefined;
    const msgArgs    = meta.args as Record<string, unknown> | undefined;

    if (subDisplay) {
      // ── Subagent-level tool ────────────────────────────────────────────────
      let saNodeId = displayToNode.get(subDisplay);
      if (!saNodeId) {
        const hashIdx  = subDisplay.lastIndexOf(" #");
        const baseName = hashIdx >= 0 ? subDisplay.slice(0, hashIdx) : subDisplay;
        const seqNum   = hashIdx >= 0 ? parseInt(subDisplay.slice(hashIdx + 2), 10) : 1;
        let matchCount = 0;
        for (const [, info] of taskCalls) {
          if (info.type === baseName) {
            matchCount++;
            if (matchCount === seqNum) { saNodeId = info.nodeId; break; }
          }
        }
        if (!saNodeId) {
          saNodeId = `sa-disp-${++autoId}`;
          const label = subDisplay.includes(" #") ? subDisplay.split(" #")[0] : subDisplay;
          nodeMap.set(saNodeId, {
            id: saNodeId, label, kind: "subagent", status: "running", toolCount: 0, seq: ++seq,
          });
          addEdge("orch", saNodeId);
        }
        displayToNode.set(subDisplay, saNodeId);
      }

      const saNode = nodeMap.get(saNodeId)!;
      const resultText = meta.result as string | undefined;

      if (msg.type === "tool_call") {
        saNode.toolCount++;
        if (!saNode.uniqueTools) saNode.uniqueTools = [];
        if (!saNode.uniqueTools.includes(msg.content)) saNode.uniqueTools.push(msg.content);
        const toolId = `tool-${saNodeId}-${tcId ?? ++autoId}`;
        nodeMap.set(toolId, {
          id: toolId, label: msg.content, kind: "tool", status: "running", toolCount: 0, args: msgArgs,
          seq: ++seq, durationMs: tcId ? durations[tcId] : undefined,
        });
        addEdge(saNodeId, toolId);
        if (tcId) tcToTool.set(tcId, toolId);
      } else if (msg.type === "tool_result") {
        const existing = tcId ? tcToTool.get(tcId) : undefined;
        if (existing) {
          const n = nodeMap.get(existing);
          if (n) {
            n.status = "done";
            if (tcId && durations[tcId] != null) n.durationMs = durations[tcId];
            const res = resultText ?? (typeof msg.content === "string" ? msg.content : undefined);
            if (res) n.resultPreview = res.slice(0, RESULT_PREVIEW_MAX);
          }
        } else {
          saNode.toolCount++;
          if (!saNode.uniqueTools) saNode.uniqueTools = [];
          if (!saNode.uniqueTools.includes(msg.content)) saNode.uniqueTools.push(msg.content);
          const toolId = `tool-${saNodeId}-${tcId ?? ++autoId}`;
          nodeMap.set(toolId, {
            id: toolId, label: msg.content, kind: "tool", status: "done", toolCount: 0,
            args: msgArgs, resultPreview: resultText?.slice(0, RESULT_PREVIEW_MAX),
            seq: ++seq, durationMs: tcId ? durations[tcId] : undefined,
          });
          addEdge(saNodeId, toolId);
          if (tcId) tcToTool.set(tcId, toolId);
        }
      }

    } else if (msg.content === "task") {
      // ── Task delegation orch → subagent ───────────────────────────────────
      const args   = (msgArgs ?? {}) as Record<string, unknown>;
      const saType = String(args.subagent_type ?? args.agent ?? "subagent");

      if (msg.type === "tool_call") {
        const nodeId = `sa-${tcId ?? ++autoId}`;
        nodeMap.set(nodeId, {
          id: nodeId, label: saType, kind: "subagent", status: "running", toolCount: 0, seq: ++seq,
        });
        addEdge("orch", nodeId);
        if (tcId) taskCalls.set(tcId, { type: saType, nodeId });
      } else if (msg.type === "tool_result") {
        const task = tcId ? taskCalls.get(tcId) : undefined;
        if (task) {
          const n = nodeMap.get(task.nodeId);
          if (n) n.status = "done";
        }
      }

    } else if (msg.type === "tool_call" && msg.content) {
      // ── Orchestrator's own direct tool ────────────────────────────────────
      const toolId = `orch-tool-${tcId ?? ++autoId}`;
      nodeMap.get("orch")!.toolCount++;
      nodeMap.set(toolId, {
        id: toolId, label: msg.content, kind: "tool", status: "running", toolCount: 0, args: msgArgs,
        seq: ++seq, durationMs: tcId ? durations[tcId] : undefined,
      });
      addEdge("orch", toolId);
      if (tcId) tcToTool.set(tcId, toolId);

    } else if (msg.type === "tool_result" && tcId) {
      // ── Orchestrator tool result ──────────────────────────────────────────
      const existing = tcToTool.get(tcId);
      if (existing) {
        const n = nodeMap.get(existing);
        if (n) {
          n.status = "done";
          if (durations[tcId] != null) n.durationMs = durations[tcId];
          const res = typeof msg.content === "string" ? msg.content : undefined;
          if (res && !n.resultPreview) n.resultPreview = res.slice(0, RESULT_PREVIEW_MAX);
        }
      }
    }
  }

  if (sessionDone) {
    for (const n of nodeMap.values()) n.status = "done";
  }

  // Disambiguate same-label subagents with "#N" suffixes
  const labelCounts = new Map<string, number>();
  for (const n of nodeMap.values()) {
    if (n.kind === "subagent") labelCounts.set(n.label, (labelCounts.get(n.label) ?? 0) + 1);
  }
  const labelSeq = new Map<string, number>();
  for (const n of nodeMap.values()) {
    if (n.kind === "subagent" && (labelCounts.get(n.label) ?? 0) > 1) {
      const seq = (labelSeq.get(n.label) ?? 0) + 1;
      labelSeq.set(n.label, seq);
      n.label = `${n.label} #${seq}`;
    }
  }

  return { nodes: [...nodeMap.values()], edges };
}

// ── Layout ────────────────────────────────────────────────────────────────────

interface LayoutResult {
  positions:   Map<string, { x: number; y: number }>;
  nodeWidths:  Map<string, number>;
  nodeHeights: Map<string, number>;
  totalWidth:  number;
  totalHeight: number;
}

function layoutGraph(nodes: GNode[], edges: GEdge[], expanded: Set<string>): LayoutResult {
  const positions   = new Map<string, { x: number; y: number }>();
  const nodeWidths  = new Map<string, number>();
  const nodeHeights = new Map<string, number>();

  const orch      = nodes.find(n => n.kind === "orchestrator");
  const subagents = nodes.filter(n => n.kind === "subagent");

  const toolsByParent = new Map<string, GNode[]>();
  for (const e of edges) {
    const child = nodes.find(n => n.id === e.to);
    if (child?.kind === "tool") {
      if (!toolsByParent.has(e.from)) toolsByParent.set(e.from, []);
      toolsByParent.get(e.from)!.push(child);
    }
  }

  const orchTools = toolsByParent.get("orch") ?? [];

  const saRows: GNode[][] = [];
  for (let i = 0; i < subagents.length; i += MAX_SA_PER_ROW) {
    saRows.push(subagents.slice(i, i + MAX_SA_PER_ROW));
  }

  // Width of a subagent's tool grid (capped at MAX_TOOLS_PER_ROW columns).
  const toolGridWidth = (tools: GNode[]) => {
    const cols = Math.min(tools.length, MAX_TOOLS_PER_ROW);
    return cols > 0 ? cols * TW + (cols - 1) * TVW : 0;
  };
  // Horizontal slot a subagent occupies — wide enough for its expanded tool
  // grid so neighbours in the same row never overlap.
  const saSlotWidth = (sa: GNode) => {
    if (!expanded.has(sa.id)) return NW;
    return Math.max(NW, toolGridWidth(toolsByParent.get(sa.id) ?? []));
  };
  const rowW = (row: GNode[]) =>
    row.reduce((sum, sa, i) => sum + saSlotWidth(sa) + (i > 0 ? HG : 0), 0);
  const orchExpanded = expanded.has("orch") || subagents.length === 0;
  const orchToolCols = Math.min(orchTools.length, MAX_TOOLS_PER_ROW);
  const orchToolGridW = orchExpanded && orchToolCols > 0 ? orchToolCols * TW + (orchToolCols - 1) * TVW : 0;
  const canvasW = Math.max(NW, orchToolGridW, ...saRows.map(rowW));

  const orchX = PAD + (canvasW - NW) / 2;
  if (orch) {
    positions.set(orch.id, { x: orchX, y: PAD });
    nodeWidths.set(orch.id, NW);
    nodeHeights.set(orch.id, NH);
  }

  let curY = PAD + NH + VG;

  // Orchestrator direct tools — collapsible like subagents. Auto-expanded when
  // there are no subagents (otherwise the canvas would be empty), else they
  // fold behind the orchestrator's count chip to reduce clutter.
  if (orchTools.length > 0 && orchExpanded) {
    const gridW = orchToolCols * TW + (orchToolCols - 1) * TVW;
    const startX = PAD + (canvasW - gridW) / 2;
    let maxBottom = curY;
    orchTools.forEach((t, i) => {
      const col = i % MAX_TOOLS_PER_ROW;
      const row = Math.floor(i / MAX_TOOLS_PER_ROW);
      const px = startX + col * (TW + TVW);
      const py = curY + row * (TH + TVH);
      positions.set(t.id, { x: px, y: py });
      nodeWidths.set(t.id, TW);
      nodeHeights.set(t.id, TH);
      const bottom = py + TH;
      if (bottom > maxBottom) maxBottom = bottom;
    });
    curY = maxBottom + VG;
  }

  // Subagent rows
  for (const rowSAs of saRows) {
    const rw = rowW(rowSAs);
    const startX = PAD + (canvasW - rw) / 2;
    let cx = startX;
    let maxColH = NH;

    for (const sa of rowSAs) {
      const slotW = saSlotWidth(sa);
      // Center the node within its (possibly wider) slot.
      positions.set(sa.id, { x: cx + (slotW - NW) / 2, y: curY });
      nodeWidths.set(sa.id, NW);
      nodeHeights.set(sa.id, NH);

      if (expanded.has(sa.id)) {
        const tools = toolsByParent.get(sa.id) ?? [];
        if (tools.length > 0) {
          const gridW  = toolGridWidth(tools);
          const toolX  = cx + (slotW - gridW) / 2;
          const toolY  = curY + NH + VG;
          tools.forEach((t, i) => {
            const col = i % MAX_TOOLS_PER_ROW;
            const row = Math.floor(i / MAX_TOOLS_PER_ROW);
            positions.set(t.id, { x: toolX + col * (TW + TVW), y: toolY + row * (TH + TVH) });
            nodeWidths.set(t.id, TW);
            nodeHeights.set(t.id, TH);
          });
          const toolRows = Math.ceil(tools.length / MAX_TOOLS_PER_ROW);
          const colH = NH + VG + toolRows * TH + (toolRows - 1) * TVH;
          if (colH > maxColH) maxColH = colH;
        }
      }
      cx += slotW + HG;
    }
    curY += maxColH + SA_ROW_GAP;
  }

  const allX = [...positions.entries()].map(([id, p]) => p.x + (nodeWidths.get(id) ?? NW));
  const allY = [...positions.entries()].map(([id, p]) => p.y + (nodeHeights.get(id) ?? NH));

  return {
    positions, nodeWidths, nodeHeights,
    totalWidth:  allX.length ? Math.max(...allX) + PAD : NW + 2 * PAD,
    totalHeight: allY.length ? Math.max(...allY) + PAD : NH + 2 * PAD,
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function argHint(args: Record<string, unknown> | undefined): string | null {
  if (!args) return null;
  for (const key of ["url", "selector", "text", "query", "path", "key", "value", "command"]) {
    const v = args[key];
    if (typeof v === "string" && v.trim()) {
      const s = v.trim();
      return s.length > 32 ? s.slice(0, 30) + "…" : s;
    }
  }
  for (const v of Object.values(args)) {
    if (typeof v === "string" && v.trim()) {
      const s = v.trim();
      return s.length > 32 ? s.slice(0, 30) + "…" : s;
    }
  }
  return null;
}

// ── Component ─────────────────────────────────────────────────────────────────

interface AgentGraphProps {
  messages: ChatMessage[];
  onClose: () => void;
  /** Render as a full-panel (no outer border/resize) — used in RunDetailPage */
  fullPanel?: boolean;
  /** tool_call_id → duration_ms, sourced from the timeline (which computes it). */
  durations?: Record<string, number>;
}

function formatGraphDuration(ms?: number): string | null {
  if (ms == null) return null;
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

const MIN_HEIGHT     = 140;
const MAX_HEIGHT     = 700;
const DEFAULT_HEIGHT = 280;
const MIN_PANEL_W    = 140;
const MAX_PANEL_W    = 400;
const DEFAULT_PANEL_W = 200;

export function AgentGraph({ messages, onClose, fullPanel = false, durations = {} }: AgentGraphProps) {
  const [expanded,    setExpanded]    = useState<Set<string>>(new Set());
  const [selectedId,  setSelectedId]  = useState<string | null>(null);
  const [height,      setHeight]      = useState(DEFAULT_HEIGHT);
  const [panelWidth,  setPanelWidth]  = useState(DEFAULT_PANEL_W);
  const [zoom,        setZoom]        = useState(1);
  const [pan,         setPan]         = useState({ x: 0, y: 0 });
  const [isDragging,  setIsDragging]  = useState(false);
  // User-dragged node positions (canvas-space), layered on top of the layout.
  const [nodeOverrides, setNodeOverrides] = useState<Record<string, { x: number; y: number }>>({});
  // A drag moves the grabbed node plus any of its tool children (so moving a
  // subagent carries its turns with it). Each member records its start origin.
  const dragNode = useRef<{ mx: number; my: number; members: { id: string; ox: number; oy: number }[] } | null>(null);

  const canvasRef  = useRef<HTMLDivElement>(null);
  // Once the user pans/zooms we stop auto-refitting, so live runs that keep
  // adding nodes don't yank the viewport out from under them.
  const userInteracted = useRef(false);
  const panOrigin  = useRef<{ mx: number; my: number; px: number; py: number } | null>(null);
  const didDrag    = useRef(false);

  // ── pointer-capture pan ───────────────────────────────────────────────────
  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    // Only start pan when clicking the canvas background, not a node
    if ((e.target as HTMLElement).closest("[data-node]")) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    panOrigin.current = { mx: e.clientX, my: e.clientY, px: pan.x, py: pan.y };
    didDrag.current = false;
    setIsDragging(true);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!panOrigin.current) return;
    const dx = e.clientX - panOrigin.current.mx;
    const dy = e.clientY - panOrigin.current.my;
    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) { didDrag.current = true; userInteracted.current = true; }
    setPan({ x: panOrigin.current.px + dx, y: panOrigin.current.py + dy });
  };

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    e.currentTarget.releasePointerCapture(e.pointerId);
    panOrigin.current = null;
    setIsDragging(false);
  };

  // ── scroll to zoom ────────────────────────────────────────────────────────
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    userInteracted.current = true;
    const step = e.deltaY > 0 ? -0.08 : 0.08;
    setZoom(z => Math.min(2.5, Math.max(0.3, parseFloat((z + step).toFixed(2)))));
  };

  // ── vertical resize handle (only in inline mode) ─────────────────────────
  const resizeOrigin = useRef<{ y: number; h: number } | null>(null);
  const onResizeDown = (e: React.MouseEvent) => {
    e.preventDefault();
    resizeOrigin.current = { y: e.clientY, h: height };
    const onMove = (ev: MouseEvent) => {
      if (!resizeOrigin.current) return;
      const delta = resizeOrigin.current.y - ev.clientY;
      setHeight(Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, resizeOrigin.current.h + delta)));
    };
    const onUp = () => {
      resizeOrigin.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // ── detail panel resize ───────────────────────────────────────────────────
  const panelResizeOrigin = useRef<{ x: number; w: number } | null>(null);
  const onPanelResizeDown = (e: React.MouseEvent) => {
    e.preventDefault();
    panelResizeOrigin.current = { x: e.clientX, w: panelWidth };
    const onMove = (ev: MouseEvent) => {
      if (!panelResizeOrigin.current) return;
      const delta = panelResizeOrigin.current.x - ev.clientX;
      setPanelWidth(Math.min(MAX_PANEL_W, Math.max(MIN_PANEL_W, panelResizeOrigin.current.w + delta)));
    };
    const onUp = () => {
      panelResizeOrigin.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // ── graph data ────────────────────────────────────────────────────────────
  const { nodes, edges } = useMemo(() => buildGraph(messages, durations), [messages, durations]);
  const nodeMap = useMemo(() => new Map(nodes.map(n => [n.id, n])), [nodes]);
  const layout  = useMemo(() => layoutGraph(nodes, edges, expanded), [nodes, edges, expanded]);
  const { positions, nodeWidths, nodeHeights, totalWidth, totalHeight } = layout;
  const selected = selectedId ? (nodeMap.get(selectedId) ?? null) : null;

  // ── fit-to-screen ─────────────────────────────────────────────────────────
  const fitToScreen = useCallback(() => {
    if (!canvasRef.current) return;
    const { clientWidth: cw, clientHeight: ch } = canvasRef.current;
    if (!cw || !ch || !totalWidth || !totalHeight) return;
    const fitZoom = Math.min(cw / totalWidth, ch / totalHeight) * 0.92;
    const clamped = Math.min(2.5, Math.max(0.3, parseFloat(fitZoom.toFixed(2))));
    const sw = totalWidth  * clamped;
    const sh = totalHeight * clamped;
    setZoom(clamped);
    setPan({ x: Math.max(0, (cw - sw) / 2), y: Math.max(0, (ch - sh) / 2) });
  }, [totalWidth, totalHeight]);

  // Auto-fit on first content and on subsequent growth (live runs), until the
  // user takes manual control of the viewport.
  useEffect(() => {
    if (userInteracted.current || !totalWidth || !totalHeight) return;
    const raf = requestAnimationFrame(() => fitToScreen());
    return () => cancelAnimationFrame(raf);
  }, [fitToScreen, totalWidth, totalHeight]);

  const subagentCount = nodes.filter(n => n.kind === "subagent").length;
  const toolCount     = nodes.filter(n => n.kind === "tool").length;

  const toggleExpand = (id: string) =>
    setExpanded(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  const visibleNodes = nodes.filter(n => positions.has(n.id));
  const visibleEdges = edges.filter(e => positions.has(e.from) && positions.has(e.to));

  if (subagentCount === 0 && toolCount === 0) return null;

  // ── node click (ignore if dragged) ────────────────────────────────────────
  const handleNodeClick = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (didDrag.current) return;
    setSelectedId(prev => prev === id ? null : id);
  };

  // ── node dragging ─────────────────────────────────────────────────────────
  // Effective (possibly user-moved) position of a node in canvas space.
  const effPos = (id: string) => nodeOverrides[id] ?? positions.get(id);

  const onNodePointerDown = (e: React.PointerEvent<HTMLDivElement>, id: string) => {
    if (e.button !== 0) return;
    // Let the expand chip handle its own clicks without starting a drag.
    if ((e.target as HTMLElement).closest("[data-expand]")) { didDrag.current = false; return; }
    e.stopPropagation();
    const cur = effPos(id);
    if (!cur) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    // Gather the grabbed node plus its tool children so they travel together.
    const memberIds = [id, ...edges.filter(ed => ed.from === id && nodeMap.get(ed.to)?.kind === "tool").map(ed => ed.to)];
    const members = memberIds
      .map(mid => { const p = effPos(mid); return p ? { id: mid, ox: p.x, oy: p.y } : null; })
      .filter((m): m is { id: string; ox: number; oy: number } => m != null);
    dragNode.current = { mx: e.clientX, my: e.clientY, members };
    didDrag.current = false;
  };

  const onNodePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = dragNode.current;
    if (!d) return;
    e.stopPropagation();
    const dx = (e.clientX - d.mx) / zoom;
    const dy = (e.clientY - d.my) / zoom;
    if (Math.abs(dx) > 2 / zoom || Math.abs(dy) > 2 / zoom) didDrag.current = true;
    setNodeOverrides(prev => {
      const next = { ...prev };
      for (const m of d.members) next[m.id] = { x: m.ox + dx, y: m.oy + dy };
      return next;
    });
  };

  const onNodePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragNode.current) return;
    e.stopPropagation();
    e.currentTarget.releasePointerCapture(e.pointerId);
    dragNode.current = null;
  };

  // Double-click a node to snap it (and its tool children) back to the layout.
  const resetNode = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    const ids = [id, ...edges.filter(ed => ed.from === id && nodeMap.get(ed.to)?.kind === "tool").map(ed => ed.to)];
    setNodeOverrides(prev => {
      if (!ids.some(i => i in prev)) return prev;
      const next = { ...prev };
      for (const i of ids) delete next[i];
      return next;
    });
  };

  const scaledW = totalWidth  * zoom;
  const scaledH = totalHeight * zoom;

  // ── outer wrapper differs for inline vs full-panel ─────────────────────────
  const outerStyle = fullPanel
    ? { display: "flex", flexDirection: "column" as const, height: "100%", minHeight: 0 }
    : { display: "flex", flexDirection: "column" as const, height, flexShrink: 0 };

  const outerCls = fullPanel
    ? "bg-th-bg-secondary"
    : "border-t border-th-border bg-th-bg-secondary";

  return (
    <div className={outerCls} style={outerStyle}>
      {/* Vertical resize handle (inline only) */}
      {!fullPanel && (
        <div
          onMouseDown={onResizeDown}
          className="h-1.5 w-full shrink-0 cursor-ns-resize flex items-center justify-center group"
          title="Drag to resize"
        >
          <div className="w-8 h-0.5 rounded-full bg-th-border group-hover:bg-blue-400/50 transition-colors" />
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-th-border/60 bg-th-bg-secondary/80 backdrop-blur-xl shrink-0">
        <div className="flex items-center gap-1.5 text-[11px] text-th-text-secondary font-medium">
          <GitBranch size={11} className="text-blue-400" />
          Agent graph
          <span className="text-[10px] text-th-text-muted/70 ml-1 tabular-nums">
            {subagentCount} subagent{subagentCount !== 1 ? "s" : ""} · {toolCount} tool{toolCount !== 1 ? "s" : ""}
          </span>
        </div>
        <div className="flex items-center gap-0.5">
          <button
            onClick={() => { userInteracted.current = true; setZoom(z => Math.max(0.3, parseFloat((z - 0.1).toFixed(1)))); }}
            className="p-1 rounded-lg text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover transition-all duration-200 active:scale-[0.92]"
            title="Zoom out (–)"
          >
            <ZoomOut size={12} />
          </button>
          <span className="text-[10px] text-th-text-muted w-9 text-center tabular-nums">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => { userInteracted.current = true; setZoom(z => Math.min(2.5, parseFloat((z + 0.1).toFixed(1)))); }}
            className="p-1 rounded-lg text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover transition-all duration-200 active:scale-[0.92]"
            title="Zoom in (+)"
          >
            <ZoomIn size={12} />
          </button>
          <button
            onClick={() => { userInteracted.current = false; fitToScreen(); }}
            className="p-1 rounded-lg text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover transition-all duration-200 active:scale-[0.92] ml-0.5"
            title="Fit to screen"
          >
            <Maximize2 size={11} />
          </button>
          {Object.keys(nodeOverrides).length > 0 && (
            <button
              onClick={() => setNodeOverrides({})}
              className="p-1 rounded-lg text-th-text-muted hover:text-blue-400 hover:bg-blue-500/10 transition-all duration-200 active:scale-[0.92]"
              title="Reset node positions"
            >
              <RotateCcw size={11} />
            </button>
          )}
          <div className="w-px h-3 bg-th-border/60 mx-1" />
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-all duration-200 active:scale-[0.92]"
            title="Close"
          >
            <X size={11} />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 min-h-0">
        {/* ── Canvas ── */}
        <div
          ref={canvasRef}
          className="flex-1 min-w-0 overflow-hidden relative select-none"
          style={{ cursor: isDragging ? "grabbing" : "grab" }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onWheel={onWheel}
        >
          {/* Hint */}
          <span className="absolute bottom-1.5 right-2.5 text-[9px] text-th-text-muted/30 pointer-events-none z-10 select-none">
            drag node to move · drag canvas to pan · scroll to zoom
          </span>

          {/* Pannable + zoomable content */}
          <div
            style={{
              position: "absolute",
              top: 0, left: 0,
              transform: `translate(${pan.x}px, ${pan.y}px)`,
              willChange: "transform",
              touchAction: "none",
            }}
          >
            {/* Inner canvas (scaled) */}
            <div style={{ position: "relative", width: scaledW, height: scaledH }}>
              {/* Edges SVG */}
              <svg
                style={{ position: "absolute", top: 0, left: 0, width: scaledW, height: scaledH, pointerEvents: "none", overflow: "visible" }}
              >
                {visibleEdges.map((e, i) => {
                  const sp = effPos(e.from);
                  const tp = effPos(e.to);
                  if (!sp || !tp) return null;
                  const sw = nodeWidths.get(e.from)  ?? NW;
                  const sh = nodeHeights.get(e.from) ?? NH;
                  const tw = nodeWidths.get(e.to)    ?? TW;
                  const sx = (sp.x + sw / 2) * zoom;
                  const sy = (sp.y + sh)      * zoom;
                  const tx = (tp.x + tw / 2)  * zoom;
                  const ty = tp.y              * zoom;
                  const my = (sy + ty) / 2;
                  const toNode  = nodeMap.get(e.to);
                  const active  = toNode?.status === "running";
                  return (
                    <path
                      key={i}
                      d={`M ${sx} ${sy} C ${sx} ${my} ${tx} ${my} ${tx} ${ty}`}
                      fill="none"
                      className={active ? "animate-dash-flow" : undefined}
                      style={{ stroke: active ? "rgb(96 165 250)" : "rgb(var(--color-border-strong))" }}
                      strokeWidth={active ? 1.5 : 1}
                      strokeOpacity={active ? 0.7 : 0.4}
                      strokeDasharray={active ? "4 3" : undefined}
                      strokeLinecap="round"
                    />
                  );
                })}
              </svg>

              {/* Nodes */}
              {visibleNodes.map(node => {
                const pos = effPos(node.id);
                if (!pos) return null;
                const nw     = (nodeWidths.get(node.id)  ?? NW) * zoom;
                const nh     = (nodeHeights.get(node.id) ?? NH) * zoom;
                const px     = pos.x * zoom;
                const py     = pos.y * zoom;
                const isSel  = selectedId === node.id;
                const isExp  = expanded.has(node.id);
                const isDone = node.status === "done";
                const hint   = node.kind === "tool" ? argHint(node.args) : null;
                const dur    = node.kind === "tool" ? formatGraphDuration(node.durationMs) : null;
                const hasTools = (node.kind === "subagent" || (node.kind === "orchestrator" && subagentCount > 0)) && node.toolCount > 0;

                // Per-kind styling — unified blue/neutral system palette.
                let border = "", bg = "";
                if (node.kind === "orchestrator") {
                  border = "border-blue-500/50"; bg = "bg-blue-500/[0.10]";
                } else if (node.kind === "subagent") {
                  border = isSel ? "border-blue-400/70" : "border-blue-500/30";
                  bg     = isSel ? "bg-blue-500/[0.10]" : "bg-th-card-bg";
                } else {
                  border = isSel  ? "border-blue-400/60"
                         : isDone ? "border-emerald-500/25"
                                  : "border-amber-400/40";
                  bg = isSel ? "bg-blue-500/[0.08]" : "bg-th-inset-bg/70";
                }

                const fs  = Math.round(10.5 * zoom);
                const fss = Math.round(9    * zoom);
                const ico = Math.round(10   * zoom);

                return (
                  <div
                    key={node.id}
                    data-node
                    style={{ position: "absolute", left: px, top: py, width: nw, height: nh, touchAction: "none" }}
                    className={`rounded-xl border flex flex-col justify-center gap-0.5 px-2 cursor-grab active:cursor-grabbing transition-colors shadow-sm shadow-black/[0.04] ${border} ${bg} hover:border-blue-400/50`}
                    onClick={ev => handleNodeClick(ev, node.id)}
                    onDoubleClick={ev => resetNode(ev, node.id)}
                    onPointerDown={ev => onNodePointerDown(ev, node.id)}
                    onPointerMove={onNodePointerMove}
                    onPointerUp={onNodePointerUp}
                    onPointerCancel={onNodePointerUp}
                    title="Drag to move · double-click to reset"
                  >
                    {/* Top row: icon + label + expand badge + status dot */}
                    <div className="flex items-center gap-1 overflow-hidden">
                      {node.kind === "tool"
                        ? (() => { const { Icon, className: c } = getToolIcon(node.label);  return <Icon size={ico} className={c + " shrink-0"} />; })()
                        : (() => { const { Icon, className: c } = getAgentIcon(node.kind); return <Icon size={ico} className={c + " shrink-0"} />; })()
                      }
                      <span className="font-medium text-th-text-primary truncate flex-1 leading-none" style={{ fontSize: fs }}>
                        {node.label}
                      </span>
                      {hasTools && (
                        <button
                          data-node
                          data-expand
                          onClick={ev => { ev.stopPropagation(); if (!didDrag.current) toggleExpand(node.id); }}
                          className="flex items-center gap-0.5 shrink-0 text-th-text-muted hover:text-th-text-secondary transition-colors rounded-full bg-th-inset-bg border border-th-border/70 px-1.5 leading-none"
                          style={{ fontSize: fss }}
                          title={isExp ? "Collapse" : "Expand tool calls"}
                        >
                          {node.toolCount}
                          {isExp
                            ? <ChevronDown size={fss} strokeWidth={2.5} />
                            : <ChevronRight size={fss} strokeWidth={2.5} />
                          }
                        </button>
                      )}
                      {/* Status dot for tool nodes */}
                      {node.kind === "tool" && (
                        <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${isDone ? "bg-emerald-400" : "bg-amber-400 animate-pulse"}`} />
                      )}
                    </div>

                    {/* Subtitle: duration + arg hint for tools */}
                    {zoom >= 0.6 && node.kind === "tool" && (dur || hint) && (
                      <span
                        className="flex items-center gap-1 leading-none pl-4 overflow-hidden"
                        style={{ fontSize: fss }}
                      >
                        {dur && <span className="text-emerald-400/90 tabular-nums shrink-0">{dur}</span>}
                        {dur && hint && <span className="text-th-text-muted/40 shrink-0">·</span>}
                        {hint && <span className="text-th-text-muted truncate">{hint}</span>}
                      </span>
                    )}
                    {zoom >= 0.65 && node.kind === "subagent" && node.uniqueTools && node.uniqueTools.length > 0 && (
                      <span
                        className="text-th-text-muted/70 truncate leading-none pl-4"
                        style={{ fontSize: fss }}
                      >
                        {[...new Set(node.uniqueTools)].slice(0, 3).join(", ")}
                        {node.uniqueTools.length > 3 ? " …" : ""}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* ── Detail panel ── */}
        {selected && (
          <div className="shrink-0 flex border-l border-th-border/60" style={{ width: panelWidth }}>
            {/* Resize handle */}
            <div
              onMouseDown={onPanelResizeDown}
              className="w-1.5 shrink-0 cursor-ew-resize flex items-center justify-center group self-stretch hover:bg-blue-400/5 transition-colors"
              title="Drag to resize"
            >
              <div className="h-8 w-0.5 rounded-full bg-th-border group-hover:bg-blue-400/50 transition-colors" />
            </div>
            <div className="flex-1 min-w-0 px-3 py-3 overflow-y-auto space-y-3">
              {/* Title */}
              <div className="flex items-start justify-between gap-1">
                <span className="text-[12px] font-semibold text-th-text-primary leading-tight break-all flex items-center gap-1.5">
                  {selected.kind === "tool"
                    ? (() => { const { Icon, className: c } = getToolIcon(selected.label);  return <Icon size={13} className={c} />; })()
                    : (() => { const { Icon, className: c } = getAgentIcon(selected.kind); return <Icon size={13} className={c} />; })()
                  }
                  {selected.label}
                </span>
                <button
                  onClick={() => setSelectedId(null)}
                  className="text-th-text-muted hover:text-th-text-primary transition-colors shrink-0 mt-0.5"
                >
                  <X size={10} />
                </button>
              </div>

              {/* Meta row */}
              <div className="flex gap-2 flex-wrap">
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-th-inset-bg border border-th-border/60 text-th-text-muted capitalize">
                  {selected.kind}
                </span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded border font-medium ${selected.status === "done" ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" : "bg-amber-500/10 border-amber-500/20 text-amber-400"}`}>
                  {selected.status === "done" ? "done" : "running"}
                </span>
                {selected.seq != null && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-th-inset-bg border border-th-border/60 text-th-text-muted tabular-nums">
                    step {selected.seq}
                  </span>
                )}
                {formatGraphDuration(selected.durationMs) && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 tabular-nums">
                    {formatGraphDuration(selected.durationMs)}
                  </span>
                )}
                {selected.toolCount > 0 && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-th-inset-bg border border-th-border/60 text-th-text-muted tabular-nums">
                    {selected.toolCount} calls
                  </span>
                )}
              </div>

              {/* Args */}
              {selected.args && Object.keys(selected.args).length > 0 && (
                <div>
                  <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-wider mb-1.5">Arguments</p>
                  <div className="rounded-lg bg-th-inset-bg border border-th-border/50 p-2 space-y-1">
                    {Object.entries(selected.args).map(([k, v]) => (
                      <div key={k} className="text-[10.5px]">
                        <span className="text-th-text-muted font-medium">{k}:</span>{" "}
                        <span className="text-th-text-secondary break-all">
                          {String(v).slice(0, 120)}{String(v).length > 120 ? "…" : ""}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Tools used (subagents) */}
              {selected.kind === "subagent" && selected.uniqueTools && selected.uniqueTools.length > 0 && (
                <div>
                  <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-wider mb-1.5">Tools used</p>
                  <div className="space-y-1">
                    {[...new Set(selected.uniqueTools)].map(t => {
                      const { Icon, className: c } = getToolIcon(t);
                      return (
                        <div key={t} className="flex items-center gap-1.5 text-[11px] text-th-text-secondary">
                          <Icon size={11} className={c} />
                          {t}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Result preview */}
              {selected.resultPreview && (
                <div>
                  <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-wider mb-1.5">Result preview</p>
                  <pre className="text-[11px] text-th-text-muted bg-th-inset-bg border border-th-border/50 rounded-lg p-2 leading-relaxed whitespace-pre-wrap break-words max-h-48 overflow-y-auto font-mono">
                    {selected.resultPreview}{selected.resultPreview.length >= RESULT_PREVIEW_MAX ? "…" : ""}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
