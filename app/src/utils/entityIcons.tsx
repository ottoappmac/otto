/**
 * Single source of truth for entity icons across Otto's UI.
 *
 * All icon helpers return a lucide-react component + a Tailwind className string.
 * Usage:
 *   const { Icon, className } = getToolIcon("execute");
 *   <Icon size={14} className={className} />
 *
 * Always pair icons with text labels or aria-label/title attributes —
 * icons alone are never the sole signal.
 */
import type { LucideIcon } from "lucide-react";
import {
  Globe,
  Terminal,
  Search,
  FileText,
  FilePen,
  FolderOpen,
  GitBranch,
  ListTodo,
  Brain,
  Calendar,
  Workflow,
  Sparkles,
  Server,
  MessageCircleQuestion,
  Image,
  Wrench,
  Bot,
  Users,
  MessageSquare,
  Zap,
  Mic,
  CheckCircle2,
  XCircle,
  Square,
  Loader2,
  Cpu,
  Cloud,
  CircleDot,
} from "lucide-react";
import type { RunStatus, TriggerSource } from "../types";

export interface IconConfig {
  Icon: LucideIcon;
  className: string;
}

// ---------------------------------------------------------------------------
// Tool icons
// ---------------------------------------------------------------------------

const TOOL_ICON_MAP: Record<string, IconConfig> = {
  // Browser / web navigation
  browser_navigate: { Icon: Globe, className: "text-blue-400" },
  browser_click: { Icon: Globe, className: "text-blue-400" },
  browser_type: { Icon: Globe, className: "text-blue-400" },
  browser_fill_form: { Icon: Globe, className: "text-blue-400" },
  browser_snapshot: { Icon: Globe, className: "text-blue-400" },
  browser_take_screenshot: { Icon: Globe, className: "text-blue-400" },
  browser_tabs: { Icon: Globe, className: "text-blue-400" },
  browser_go_back: { Icon: Globe, className: "text-blue-400" },
  browser_go_forward: { Icon: Globe, className: "text-blue-400" },
  browser_select_option: { Icon: Globe, className: "text-blue-400" },
  browser_hover: { Icon: Globe, className: "text-blue-400" },
  browser_drag: { Icon: Globe, className: "text-blue-400" },
  browser_press_key: { Icon: Globe, className: "text-blue-400" },
  browser_wait: { Icon: Globe, className: "text-blue-400" },
  browser_close: { Icon: Globe, className: "text-blue-400" },
  // Shell / execution
  execute: { Icon: Terminal, className: "text-th-text-secondary" },
  // Research / search
  web_research: { Icon: Search, className: "text-sky-400" },
  doc_research: { Icon: Search, className: "text-sky-400" },
  wikipedia_search: { Icon: Search, className: "text-sky-400" },
  duckduckgo_search: { Icon: Search, className: "text-sky-400" },
  grep: { Icon: Search, className: "text-sky-400" },
  // File reads
  read_file: { Icon: FileText, className: "text-th-text-secondary" },
  doc_reader: { Icon: FileText, className: "text-th-text-secondary" },
  // File writes / edits
  write_file: { Icon: FilePen, className: "text-emerald-400" },
  edit_file: { Icon: FilePen, className: "text-emerald-400" },
  // File listing
  ls: { Icon: FolderOpen, className: "text-th-text-muted" },
  glob: { Icon: FolderOpen, className: "text-th-text-muted" },
  // Subagent delegation
  task: { Icon: GitBranch, className: "text-violet-400" },
  // Task management
  write_todos: { Icon: ListTodo, className: "text-blue-400" },
  // Memory
  memory_search: { Icon: Brain, className: "text-pink-400" },
  memory_store: { Icon: Brain, className: "text-pink-400" },
  // Schedules
  list_schedules: { Icon: Calendar, className: "text-blue-400" },
  create_schedule: { Icon: Calendar, className: "text-blue-400" },
  update_schedule: { Icon: Calendar, className: "text-blue-400" },
  delete_schedule: { Icon: Calendar, className: "text-blue-400" },
  run_schedule_now: { Icon: Calendar, className: "text-blue-400" },
  stop_schedule: { Icon: Calendar, className: "text-blue-400" },
  // Agent / skill management
  create_agent_config: { Icon: Workflow, className: "text-violet-400" },
  update_agent_config: { Icon: Workflow, className: "text-violet-400" },
  list_existing_agents: { Icon: Workflow, className: "text-violet-400" },
  create_skill: { Icon: Sparkles, className: "text-violet-400" },
  update_skill: { Icon: Sparkles, className: "text-violet-400" },
  list_existing_skills: { Icon: Sparkles, className: "text-violet-400" },
  // MCP / tools discovery
  list_available_mcp_servers: { Icon: Server, className: "text-th-text-secondary" },
  discover_available_tools: { Icon: Server, className: "text-th-text-secondary" },
  // User interaction
  ask_user: { Icon: MessageCircleQuestion, className: "text-blue-400" },
  // Images
  view_image: { Icon: Image, className: "text-blue-400" },
};

export function getToolIcon(name: string): IconConfig {
  if (name in TOOL_ICON_MAP) return TOOL_ICON_MAP[name];
  // Prefix matching for browser_* etc.
  if (name.startsWith("browser_")) return { Icon: Globe, className: "text-blue-400" };
  if (name.startsWith("memory_")) return { Icon: Brain, className: "text-pink-400" };
  if (name.includes("search") || name.includes("research")) {
    return { Icon: Search, className: "text-sky-400" };
  }
  return { Icon: Wrench, className: "text-th-text-muted" };
}

// ---------------------------------------------------------------------------
// Agent / node icons
// ---------------------------------------------------------------------------

export function getAgentIcon(kind: "orchestrator" | "subagent" | "tool" | string): IconConfig {
  switch (kind) {
    case "orchestrator":
      return { Icon: Bot, className: "text-blue-400" };
    case "subagent":
      return { Icon: Users, className: "text-violet-400" };
    default:
      return { Icon: Workflow, className: "text-th-text-muted" };
  }
}

// ---------------------------------------------------------------------------
// Trigger source icons
// ---------------------------------------------------------------------------

export function getSourceIcon(source: TriggerSource): IconConfig {
  switch (source) {
    case "schedule":
      return { Icon: Calendar, className: "text-blue-400" };
    case "trigger":
      return { Icon: Zap, className: "text-amber-400" };
    case "ambient":
      return { Icon: Sparkles, className: "text-violet-400" };
    case "voice":
      return { Icon: Mic, className: "text-emerald-400" };
    case "spawn":
      return { Icon: GitBranch, className: "text-blue-400" };
    default:
      return { Icon: MessageSquare, className: "text-th-text-muted" };
  }
}

export function getSourceLabel(source: TriggerSource): string {
  switch (source) {
    case "schedule": return "Schedule";
    case "trigger": return "Trigger";
    case "ambient": return "Ambient";
    case "voice": return "Voice";
    case "spawn": return "Spawned";
    case "claude-hook": return "Claude Hook";
    default: return "Manual";
  }
}

// ---------------------------------------------------------------------------
// Run status icons
// ---------------------------------------------------------------------------

export function getStatusIcon(status: RunStatus | string): IconConfig {
  switch (status) {
    case "running":
      return { Icon: Loader2, className: "text-th-text-muted animate-spin" };
    case "completed":
      return { Icon: CheckCircle2, className: "text-emerald-400" };
    case "error":
      return { Icon: XCircle, className: "text-red-400" };
    case "stopped":
      return { Icon: Square, className: "text-th-text-muted" };
    case "awaiting_input":
      return { Icon: MessageCircleQuestion, className: "text-blue-400" };
    default:
      return { Icon: CircleDot, className: "text-th-text-muted/40" };
  }
}

export function getStatusLabel(status: RunStatus | string): string {
  switch (status) {
    case "running": return "Running";
    case "completed": return "Completed";
    case "error": return "Error";
    case "stopped": return "Stopped";
    case "awaiting_input": return "Awaiting input";
    default: return "Idle";
  }
}

export function getStatusColor(status: RunStatus | string): string {
  switch (status) {
    case "running": return "text-th-text-muted";
    case "completed": return "text-emerald-400";
    case "error": return "text-red-400";
    case "stopped": return "text-th-text-muted";
    case "awaiting_input": return "text-blue-400";
    default: return "text-th-text-muted/40";
  }
}

export function getStatusBorderColor(status: RunStatus | string): string {
  switch (status) {
    case "running": return "border-emerald-500/30";
    case "completed": return "border-emerald-500/20";
    case "error": return "border-red-500/30";
    case "stopped": return "border-th-border";
    case "awaiting_input": return "border-blue-400/30";
    default: return "border-th-card-border";
  }
}

// ---------------------------------------------------------------------------
// Provider / LLM family icons
// ---------------------------------------------------------------------------

export function getProviderIcon(family: string | null | undefined): IconConfig {
  switch (family) {
    case "mlx":
    case "omlx":
    case "exo":
    case "afm":
      return { Icon: Cpu, className: "text-emerald-400" };
    case "anthropic":
    case "openai":
    case "cohere":
    case "frontier":
      return { Icon: Cloud, className: "text-sky-400" };
    default:
      return { Icon: Cpu, className: "text-th-text-muted" };
  }
}
