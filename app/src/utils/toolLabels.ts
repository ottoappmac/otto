const TOOL_LABELS: Record<string, string> = {
  // Playwright browser
  browser_navigate: "Navigate to URL",
  browser_click: "Click element",
  browser_type: "Type text",
  browser_fill_form: "Fill form",
  browser_snapshot: "Capture page snapshot",
  browser_take_screenshot: "Take screenshot",
  browser_tabs: "Manage browser tabs",
  browser_go_back: "Go back",
  browser_go_forward: "Go forward",
  browser_select_option: "Select option",
  browser_hover: "Hover element",
  browser_drag: "Drag element",
  browser_press_key: "Press key",
  browser_wait: "Wait",
  browser_close: "Close browser",
  // Subagents
  task: "Delegate to agent",
  // Built-in file / shell
  write_file: "Write file",
  read_file: "Read file",
  edit_file: "Edit file",
  ls: "List directory",
  glob: "Find files",
  grep: "Search in files",
  execute: "Run command",
  write_todos: "Update task list",
  // Research
  web_research: "Search the web",
  doc_research: "Search document",
  doc_reader: "Read & summarise document",
  wikipedia_search: "Search Wikipedia",
  duckduckgo_search: "Web search",
  youtube_search: "Find YouTube videos",
  youtube_transcript: "Search YouTube transcript",
  // Agent management
  discover_available_tools: "Discover tools",
  list_available_mcp_servers: "List MCP servers",
  list_existing_skills: "List skills",
  list_existing_agents: "List agents",
  create_skill: "Create skill",
  create_agent_config: "Create agent",
  update_skill: "Update skill",
  update_agent_config: "Update agent",
  // Schedules
  list_schedules: "List schedules",
  create_schedule: "Create schedule",
  update_schedule: "Update schedule",
  delete_schedule: "Delete schedule",
  run_schedule_now: "Run schedule now",
  stop_schedule: "Stop schedule",
  // Session
  view_image: "View image",
  // User interaction
  ask_user: "Ask user",
};

export function getToolLabel(name: string, args?: Record<string, unknown>): string {
  const base = TOOL_LABELS[name];
  if (!args) return base ?? name.replace(/_/g, " ");
  switch (name) {
    case "browser_navigate": {
      const url = args.url as string | undefined;
      if (url) return `Navigate to ${url.length > 50 ? url.slice(0, 50) + "…" : url}`;
      break;
    }
    case "task": {
      const agent = args.subagent_type as string | undefined;
      if (agent) return `Delegate to ${agent}`;
      break;
    }
    case "execute": {
      const cmd = String(args.command ?? "").trim();
      if (cmd) return `Run \u2068${cmd.length > 50 ? cmd.slice(0, 50) + "…" : cmd}\u2069`;
      break;
    }
    case "write_file":
    case "read_file":
    case "edit_file": {
      const p = (args.path ?? args.file_path) as string | undefined;
      if (p && base) return `${base} ${p}`;
      break;
    }
    case "web_research":
    case "doc_research":
    case "wikipedia_search":
    case "youtube_search":
    case "duckduckgo_search": {
      const q = (args.query ?? args.search_query) as string | undefined;
      if (q && base) return `${base}: ${q.length > 50 ? q.slice(0, 50) + "…" : q}`;
      break;
    }
    case "youtube_transcript": {
      const q = (args.query || args.video) as string | undefined;
      if (q && base) return `${base}: ${q.length > 50 ? q.slice(0, 50) + "…" : q}`;
      break;
    }
  }
  return base ?? name.replace(/_/g, " ");
}
