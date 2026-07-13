// ---------------------------------------------------------------------------
// WebSocket message types
// ---------------------------------------------------------------------------

export type WSMessageType =
  | "user"
  | "agent"
  | "tool_call"
  | "tool_result"
  | "execute_output"
  | "hitl_request"
  | "ask_user"
  | "memory_context"
  | "memory_search"
  | "status"
  | "error"
  | "done"
  | "stopped"
  | "context_received";

export interface WSMessage {
  type: WSMessageType;
  content: string;
  metadata?: Record<string, unknown>;
  timestamp?: string;
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export interface ChatMessage {
  id: string;
  type: WSMessageType;
  content: string;
  metadata?: Record<string, unknown>;
  timestamp: Date;
  /** Which backend session this message belongs to. Used to gate display. */
  sessionId?: string;
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

export interface AgentSpec {
  name: string;
  description: string;
  system_prompt: string;
  tools: string[];
  skills: string[];
  model_override: string | null;
  /** When set: inherit | frontier | mlx | custom (library agents run as ``task`` subagents). */
  subagent_llm_family?: string | null;
  /** HF repo id for MLX family; optional (defaults to global MLX text model). */
  mlx_model_id?: string | null;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
  /** True for agents shipped with the app — managed by seed_defaults, cannot be deleted. */
  builtin?: boolean;
}

// ---------------------------------------------------------------------------
// Skill
// ---------------------------------------------------------------------------

export interface SkillSpec {
  name: string;
  description: string;
  content: string;
  created_at: string;
  updated_at: string;
  /** True for skills shipped with the app — managed by seed_defaults, cannot be deleted. */
  builtin?: boolean;
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

export interface MCPServerConfig {
  id: string;
  name: string;
  transport: string;
  url: string | null;
  command: string | null;
  args: string[];
  env: Record<string, string>;
  enabled: boolean;
  auto_start: boolean;
  excluded_tools: string[];
  builtin: boolean;
}

/**
 * Per-server interactive-auth snapshot.  Mirrors backend ``MCPAuthStatus``.
 *
 * ``kind === "static"`` means the historical paste-a-string flow —
 * the credentials dialog renders a list of text inputs.  Anything
 * else means an interactive provider (OAuth device, OAuth auth-code,
 * browser bearer-token capture) and the dialog renders a "Login"
 * button instead.
 *
 * Only booleans + names + an expiry timestamp travel over the wire —
 * never the token itself.
 */
export interface MCPAuthStatus {
  kind: "static" | "oauth_device" | "oauth_authcode" | "browser_capture" | string;
  has_bundle: boolean;
  expired: boolean;
  needs_login: boolean;
  expiry_iso: string | null;
}

export interface MCPServerStatus {
  id: string;
  name: string;
  connected: boolean;
  tool_count: number;
  tools: string[];
  excluded_tools: string[];
  error: string | null;
  auto_start: boolean;
  process_running: boolean;
  transport: string;
  url: string | null;
  port: number | null;
  command: string | null;
  args: string[];
  builtin: boolean;
  requires_os: string | null;
  os_supported: boolean;
  server_type: string;
  context_cache_active: boolean;
  // Agent-built MCP plumbing.  ``generated=true`` means the server was
  // authored at runtime by the mcp_builder pipeline; the UI uses it to
  // (a) badge the row and (b) route delete through the registry which
  // also wipes the on-disk source file plus vault credentials.
  // ``required_secrets`` lists every credential the subprocess needs in
  // env at spawn time; ``missing_secrets`` is the subset not yet stored
  // in the OS keychain — the UI shows a "Set credentials" affordance
  // when this is non-empty instead of letting Start fail.
  generated: boolean;
  required_secrets: string[];
  missing_secrets: string[];
  // Optional credentials show up in the credentials dialog (so the
  // user can set / update them) but never block the Start button —
  // the MCP has a working default for these.
  optional_secrets?: string[];
  // Interactive auth-flow status.  Defaults to ``{ kind: "static", ... }``
  // for legacy / paste-a-string MCPs so the frontend can read this
  // field unconditionally without null-checking.
  auth?: MCPAuthStatus;
}

// ---------------------------------------------------------------------------
// MLX generation stats (token throughput telemetry)
// ---------------------------------------------------------------------------

/**
 * Per-turn MLX generation telemetry forwarded on agent / tool_call message
 * metadata (`metadata.stats`). Populated only for the in-process `mlx`
 * provider; omlx/cloud turns omit these fields.
 */
export interface MlxStats {
  tokens_from_cache?: number;
  tokens_prefilled?: number;
  cache_hit_ratio?: number;
  prompt_tps?: number;
  generation_tokens?: number;
  generation_tps?: number;
  peak_memory_gb?: number;
}

/** Aggregate, session-level throughput fields shared by SessionInfo/RunInfo. */
export interface ThroughputStats {
  /** "TIPS" — token-weighted average prefill throughput (tokens/sec). */
  avg_prefill_tps?: number | null;
  /** "TOPS" — token-weighted average generation throughput (tokens/sec). */
  avg_generation_tps?: number | null;
  /** KV cache reuse ratio (0..1). */
  cache_hit_ratio?: number | null;
  /** Peak GPU memory observed during the session (GB). */
  peak_memory_gb?: number | null;
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

export interface SessionInfo extends ThroughputStats {
  id: string;
  agent_name: string | null;
  title: string;
  message_count: number;
  tools_used: string[];
  schedule_id: string | null;
  trigger_source: string | null;
  /** Set when the session was spawned by a custom trigger (trigger_source === "trigger"). */
  trigger_id?: string | null;
  /**
   * Set when the orchestrator handed off via ``spawn_followup_session``;
   * the chat header renders a "← from parent" link so the user can jump
   * back to the originating session.
   */
  parent_session_id: string | null;
  /** 0 = root session; +1 per spawn-chain hop. Capped backend-side. */
  chain_depth: number;
  created_at: string;
  updated_at: string;
  // Run metrics
  status?: RunStatus;
  finished_at?: string | null;
  duration_ms?: number | null;
  llm_provider?: string | null;
  model?: string | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  estimated_cost_usd?: number | null;
  error?: string | null;
}

// ---------------------------------------------------------------------------
// Unified Run types (dashboard)
// ---------------------------------------------------------------------------

export type RunStatus = "idle" | "running" | "completed" | "error" | "stopped" | "awaiting_input";
export type RunKind = "session" | "schedule_run" | "trigger_run";
export type TriggerSource = "schedule" | "trigger" | "ambient" | "voice" | "spawn" | "claude-hook" | null;

export interface RunInfo extends ThroughputStats {
  id: string;
  kind: RunKind;
  title: string;
  agent_name: string | null;
  trigger_source: TriggerSource;
  schedule_id: string | null;
  trigger_id: string | null;
  parent_session_id: string | null;
  chain_depth: number;
  status: RunStatus;
  started_at: string;
  finished_at: string | null;
  duration_ms: number | null;
  message_count: number;
  /** Total agent steps (model turns + tool calls), including subagents. */
  step_count: number;
  tools_used: string[];
  llm_provider: string | null;
  model: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  estimated_cost_usd: number | null;
  error: string | null;
  session_id: string | null;
  /** End-of-run evaluation summary. none | running | done | skipped | error. */
  eval_status?: string | null;
  eval_overall_score?: number | null;
  eval_pass_count?: number | null;
  eval_total?: number | null;
}

export interface EvalMetricResult {
  evaluator_type: string;
  score?: number;
  success?: boolean;
  reason?: string | null;
  threshold?: number;
  error?: string;
}

/** A single step in the evaluator's live trace, rendered like a chat turn. */
export interface EvalStep {
  /** status | thought | selection | metric_start | metric_result | done */
  kind: string;
  text?: string;
  metric?: string;
  metrics?: string[];
  score?: number | null;
  success?: boolean;
  ts?: string;
}

export interface EvalTurn {
  input: string;
  output: string;
  tools: string[];
}

export interface RunEvaluation {
  session_id: string;
  /** none | running | done | skipped | error */
  status: string;
  /** "error_analysis" for failed-run diagnosis; absent for normal metric evals. */
  kind?: string;
  manual?: boolean;
  model?: string | null;
  reason?: string;
  error?: string;
  /** Coarse failure classification for an error_analysis run. */
  error_code?: string | null;
  /** LLM's diagnosis of the failure (error_analysis runs). */
  diagnosis?: string | null;
  selected_metrics?: string[];
  steps?: EvalStep[];
  turns?: EvalTurn[];
  results: EvalMetricResult[];
  overall_score?: number | null;
  pass_count?: number;
  total?: number;
  started_at?: string;
  evaluated_at?: string;
  /** LLM-suggested improved prompt when the run scored below threshold. */
  suggested_prompt?: string | null;
  suggestion_reason?: string | null;
}

export interface RunsResponse {
  total: number;
  offset: number;
  limit: number;
  runs: RunInfo[];
}

export interface RunStats extends ThroughputStats {
  period: string;
  running_now: number;
  total_period: number;
  status_counts: Record<string, number>;
  success_rate: number;
  avg_duration_ms: number | null;
  total_steps: number;
  avg_steps_per_run: number | null;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number;
  /** Number of runs scored by the end-of-run evaluator in this period. */
  eval_count: number;
  /** Mean overall evaluation score (0..1), or null when nothing was evaluated. */
  eval_avg_score: number | null;
  /** Percentage of scored metrics that met their threshold, or null. */
  eval_pass_rate: number | null;
  top_agents: { agent: string; count: number }[];
  top_tools: { tool: string; count: number }[];
  source_breakdown: { source: string; count: number }[];
  model_breakdown: { model: string; count: number }[];
  model_cost_breakdown: { model: string; cost_usd: number }[];
  time_series: {
    ts: string;
    total: number;
    completed: number;
    error: number;
    running: number;
    eval_avg_score: number | null;
    eval_count: number;
  }[];
}

export interface TimelineEvent {
  ts?: string;
  type: string;
  role?: string;
  tool?: string;
  tool_call_id?: string;
  content: string | unknown;
  meta?: Record<string, unknown>;
  args?: Record<string, unknown>;
  subagent?: string;
  images?: { base64: string; mime_type: string }[];
  stats?: Record<string, unknown>;
  duration_ms?: number;
}

export interface SessionTimeline {
  session_id: string;
  events: TimelineEvent[];
  total: number;
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

export const MAX_SCHEDULES = 10;

export const CRON_PRESETS = [
  { id: "hourly", label: "Every hour", cron: "0 * * * *" },
  { id: "every-2h", label: "Every 2 hours", cron: "0 */2 * * *" },
  { id: "daily-9am", label: "Daily at 9:00 AM", cron: "0 9 * * *" },
  { id: "weekdays-9am", label: "Weekdays at 9:00 AM", cron: "0 9 * * 1-5" },
  { id: "weekly-mon", label: "Weekly on Monday", cron: "0 9 * * 1" },
  { id: "custom", label: "Custom", cron: null },
] as const;

export interface ScheduleSpec {
  id: string;
  agent_name: string | null;
  prompt: string;
  cron_expression: string;
  enabled: boolean;
  keep_last_n_runs: number;
  last_run: string | null;
  last_status: string | null;
  last_error: string | null;
  created_at: string;
  updated_at: string;
}

export interface ScheduleRun {
  id: string;
  schedule_id: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  message_count: number;
  error: string | null;
  session_id: string | null;
  eval_status?: string | null;
  eval_overall_score?: number | null;
  eval_pass_count?: number | null;
  eval_total?: number | null;
}

export interface ScheduleRunsResponse {
  runs: ScheduleRun[];
  total: number;
}

export interface ScheduleRunStats {
  total: number;
  success_rate: number;
  avg_duration_ms: number | null;
  total_steps: number;
  avg_steps_per_run: number | null;
  running_now: number;
}

export interface ScheduleAttachment {
  path: string;
  size: number;
}

export interface ScheduleStatusPoll {
  running: string[];
  recently_completed: {
    id: string;
    status: string;
    last_run: string | null;
    error: string | null;
  }[];
}

// ---------------------------------------------------------------------------
// Custom Triggers
// ---------------------------------------------------------------------------

export const MAX_TRIGGERS = 5;
export const MIN_POLL_SECONDS = 5;
export const MAX_POLL_SECONDS = 24 * 60 * 60;

export type TriggerType = "fileos" | "macostool" | "http" | "git" | "shell";
export type FileOsWatch = "mtime" | "size" | "exists" | "new_files";
export type OsaLanguage = "AppleScript" | "JavaScript";
export type HttpMode = "status_change" | "body_hash" | "json_value" | "regex";
export type HttpMethod = "GET" | "POST" | "HEAD";
export type ShellMode = "stdout_change" | "regex" | "exit_code_change";

export interface TriggerSpec {
  id: string;
  type: TriggerType;
  poll_seconds: number;
  agent_name: string | null;
  prompt: string;
  enabled: boolean;
  /** True for triggers seeded from the managed catalog — cannot be deleted. */
  builtin?: boolean;
  // fileos
  path: string | null;
  watch: FileOsWatch;
  glob: string | null;
  // macostool
  script: string | null;
  language: OsaLanguage;
  match: string | null;
  // http
  url: string | null;
  http_mode: HttpMode;
  method: HttpMethod;
  headers: Record<string, string>;
  body: string | null;
  json_path: string | null;
  // git
  repo_path: string | null;
  branch: string;
  author_filter: string | null;
  path_filter: string | null;
  // shell
  command: string | null;
  shell_mode: ShellMode;
  cwd: string | null;
  env: Record<string, string>;
  state_json: Record<string, unknown>;
  keep_last_n_runs: number;
  timeout_seconds: number;
  last_run: string | null;
  last_status: string | null;
  last_error: string | null;
  last_event: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface TriggerCreateRequest {
  id: string;
  type: TriggerType;
  prompt: string;
  poll_seconds?: number;
  agent_name?: string | null;
  path?: string | null;
  watch?: FileOsWatch;
  glob?: string | null;
  script?: string | null;
  language?: OsaLanguage;
  match?: string | null;
  url?: string | null;
  http_mode?: HttpMode;
  method?: HttpMethod;
  headers?: Record<string, string>;
  body?: string | null;
  json_path?: string | null;
  repo_path?: string | null;
  branch?: string;
  author_filter?: string | null;
  path_filter?: string | null;
  command?: string | null;
  shell_mode?: ShellMode;
  cwd?: string | null;
  env?: Record<string, string>;
}

export interface TriggerUpdateRequest {
  prompt?: string | null;
  poll_seconds?: number | null;
  enabled?: boolean | null;
  agent_name?: string | null;
  path?: string | null;
  watch?: FileOsWatch | null;
  glob?: string | null;
  script?: string | null;
  language?: OsaLanguage | null;
  match?: string | null;
  url?: string | null;
  http_mode?: HttpMode | null;
  method?: HttpMethod | null;
  headers?: Record<string, string> | null;
  body?: string | null;
  json_path?: string | null;
  repo_path?: string | null;
  branch?: string | null;
  author_filter?: string | null;
  path_filter?: string | null;
  command?: string | null;
  shell_mode?: ShellMode | null;
  cwd?: string | null;
  env?: Record<string, string> | null;
  keep_last_n_runs?: number | null;
}

export interface TriggerRun {
  id: string;
  trigger_id: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  message_count: number;
  error: string | null;
  session_id: string | null;
  event_payload: Record<string, unknown>;
}

export interface TriggerRunsResponse {
  runs: TriggerRun[];
  total: number;
}

export interface TriggerStatusPoll {
  running: string[];
  recently_completed: {
    id: string;
    status: string;
    last_run: string | null;
    error: string | null;
  }[];
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export interface AnthropicConfig {
  model_provider: string;
  api_key: string;
  model_name: string;
  bedrock_region: string;
  bedrock_auth_mode: string;
  aws_access_key_id: string;
  aws_secret_access_key: string;
  max_tokens: number;
  thinking_enabled: boolean;
  thinking_budget: number;
  tool_efficient: boolean;
}

export interface OpenAIConfig {
  /** "openai" for native API, "azure" for Azure OpenAI Service */
  model_provider: string;
  /** Native OpenAI API key (sk-...) */
  api_key: string;
  /** Azure OpenAI API key — stored separately so switching modes doesn't overwrite the other key */
  azure_api_key: string;
  model_name: string;
  azure_endpoint: string;
  azure_api_version: string;
  azure_deployment: string;
  max_tokens: number;
  temperature: number;
}

/** Optional friendly name for a cached Hub repo (shown in pickers). */
export interface MlxBookmark {
  repo_id: string;
  label: string;
}

/** Live GPU + RAM snapshot returned by ``GET /api/mlx/live-stats``.
 *  RAM fields mirror Activity Monitor: used = App + Wired + Compressed. */
export interface MlxLiveStats {
  apple_silicon: boolean;
  /** Bytes held by the MLX Metal allocator (GB). 0 when MLX is not yet loaded. */
  active_gpu_mem_gb: number;
  /** Apple-Silicon wired-memory ceiling (GB). Equals ram_gb on non-Apple hosts. */
  gpu_limit_gb: number;
  /** Physical RAM total (GB). */
  ram_gb: number;
  /** App Memory + Wired + Compressed — matches Activity Monitor's "Used" definition. */
  ram_used_gb: number;
  /** Active app memory (GB). */
  ram_app_gb: number;
  /** Wired (kernel + GPU locked) memory (GB). */
  ram_wired_gb: number;
  /** Compressed memory (GB). */
  ram_compressed_gb: number;
  /** Truly free pages (GB) — usually small on macOS due to disk-cache speculative use. */
  ram_free_gb: number;
  /** macOS memory pressure level — mirrors Activity Monitor's graph colour. */
  memory_pressure: "normal" | "warning" | "critical";
}

/** Local hardware probe used by the On-Device empty state and (later)
 *  the Setup Hub.  Returned by ``GET /api/mlx/capabilities``.  All
 *  numbers are GB. */
export interface MlxCapabilities {
  platform: string;
  arch: string;
  apple_silicon: boolean;
  chip: string;
  cpu_brand: string;
  ram_gb: number;
  free_disk_gb: number;
  /** Apple-Silicon GPU wired-memory ceiling.  On non-Apple hosts this
   *  equals ``ram_gb``. */
  wired_limit_gb: number;
  hf_token_set: boolean;
  hub_cache_dir: string;
  models_cached: number;
  models_cached_size_gb: number;
}

/** One scored row from ``GET /api/mlx/catalog``.  Static fields come
 *  from the curated catalog in ``backend/mlx_catalog.py``; the
 *  ``fits`` / footprint fields are computed against the live
 *  capabilities probe so they shift when the user moves the
 *  ``ctx_len`` slider. */
export interface MlxCatalogRow {
  repo_id: string;
  family: string;
  display_name: string;
  blurb: string;
  weights_gb: number;
  params_b: number;
  quant: string;
  role: string[];
  capability_tags: string[];
  n_layers: number;
  n_kv_heads: number;
  head_dim: number;
  max_position: number;
  requires_token: boolean;
  tier: "starter" | "balanced" | "power";
  featured: boolean;
  downloads: number;
  last_modified: string;
  // Score fields — only present when fetched via /api/mlx/catalog.
  kv_gb: number;
  overhead_gb: number;
  draft_gb: number;
  total_gb: number;
  fits: "comfortable" | "tight" | "over" | "unknown";
  headroom_gb: number;
  ceiling_gb: number;
  disk_ok: boolean;
  already_cached: boolean;
  /** True only when all safetensors blobs in the cached snapshot exist on
   *  disk.  False when a previous download was interrupted mid-way. */
  cache_complete: boolean;
}

export interface MlxCatalogResponse {
  capabilities: MlxCapabilities;
  ctx_len: number;
  kv_bits: number | null;
  comfortable_fraction: number;
  counts: { comfortable: number; tight: number; over: number; unknown: number };
  models: MlxCatalogRow[];
  /** True while the backend is fetching the full mlx-community listing in
   *  the background.  The UI should auto-poll until this becomes false. */
  enriching?: boolean;
}

/** Live progress for a single MLX download job. */
export interface MlxDownloadJob {
  job_id: string;
  status: "pending" | "running" | "cancelling" | "done" | "error" | "cancelled";
  message: string;
  repo_id: string;
  hub_cache: string;
  started_at: number | null;
  bytes_done: number;
  bytes_total: number;
  files_done: number;
  files_total: number;
  current_file: string;
  current_file_total: number;
  rate_bps: number;
  eta_seconds: number | null;
  use_hf_transfer: boolean;
  max_workers: number;
}

/** Hugging Face Hub repo IDs for local MLX (when LLM provider is ``mlx``). */
export interface MlxHfConfig {
  hf_llm_model_id: string;
  hf_vlm_model_id: string;
  hf_draft_llm_model_id: string;
  hf_token: string;
  /** Hub cache directory; empty = Hugging Face default (~/.cache/huggingface/hub). */
  hf_hub_cache: string;
  mlx_bookmarks: MlxBookmark[];
  /** Maps to env MLX_* — applied when provider is MLX. */
  mlx_max_tokens: number;
  mlx_temp: number;
  mlx_verbose: boolean;
  mlx_thinking: boolean;
  mlx_prompt_cache: boolean;
  mlx_system_prompt_cache: boolean;
  mlx_kv_bits: number | null;
  mlx_kv_group_size: number;
  mlx_repetition_penalty: number;
  /**
   * Soft cap on the KV prefix cache, in tokens.  When the cumulative cache
   * offset exceeds this value after a generation, the cache is trimmed (or
   * fully rebuilt for non-trimmable layer types) so long autonomous sessions
   * don't OOM the host.  ``0`` disables the cap (legacy unbounded
   * behaviour).  Default 32 768 tokens ≈ 1 GB on a 7B 4-bit model.
   */
  mlx_prompt_cache_max_tokens: number;
  /**
   * Turbo mode (oMLX-derived optimisations).  ``off`` keeps the classic
   * MLX path; implemented levels:
   *
   *   * ``basic`` — single-threaded MLX executor.
   *   * ``cache`` — basic + cross-session KV prefix sharing via a
   *     per-model singleton TurboMLXChat.
   *   * ``ssd`` — cache + disk-backed cold tier so the KV prefix
   *     survives process restarts (``turbo_ssd_dir`` /
   *     ``turbo_ssd_max_gb`` control location and budget).
   *
   * The Settings UI surfaces only two knobs — a master "Enable turbo
   * mode" toggle (off ↔ cache) and an SSD toggle (cache ↔ ssd) — but
   * the field is still the single source of truth.  Anything reading
   * this config (backend, env bridge, tests) should branch on the
   * string value, not reconstruct the UI state.  ``basic`` is reachable
   * via ``MLX_TURBO_LEVEL`` env var for diagnostics but isn't exposed
   * in the UI.  Higher levels (TurboQuant) are planned but not yet
   * wired up.  The backend factory falls back to classic on any init
   * failure, so flipping this value is always safe.
   */
  turbo_level: "off" | "basic" | "cache" | "ssd";
  /** SSD cold-tier cache directory; empty ⇒ default under <app_data>/kv_cache. */
  turbo_ssd_dir: string;
  /** Soft cap on the SSD cache footprint (GB); LRU-evicted beyond this. */
  turbo_ssd_max_gb: number;
  /** Reserved for upcoming TurboQuant level; unused today. */
  turbo_tq_bits: number;
  /** Paged-cache block size; unused until the paged allocator lands. */
  turbo_block_size: number;
}

export interface LLMConfig {
  provider: string;
  anthropic: AnthropicConfig;
  openai: OpenAIConfig;
  mlx: MlxHfConfig;
}

export interface LangSmithConfig {
  enabled: boolean;
  api_key: string;
  endpoint: string;
  project: string;
}

export interface ObservabilityConfig {
  langsmith: LangSmithConfig;
  log_level: string;
}

export interface HookDefinition {
  id: string;
  event: string;
  type: "http" | "command" | "prompt";
  url: string;
  command: string;
  prompt: string;
  matcher: string;
  timeout: number;
  enabled: boolean;
}

export interface ClaudeHookConfig {
  enabled: boolean;
  http_hooks_enabled: boolean;
  quality_gate_enabled: boolean;
  quality_gate_threshold: number;
  auto_monitor_enabled: boolean;
  max_auto_sessions: number;
  auto_monitor_agent: string;
  hooks: HookDefinition[];
}

export interface EvaluationConfig {
  /** Auto-evaluate every completed run. When true, the manual Evaluate button is disabled. */
  auto_evaluate: boolean;
  /** Analyze runs that end in error and suggest a prompt fix when applicable. */
  analyze_errors: boolean;
  /** follow_main | frontier | mlx */
  llm_family: string;
  max_metrics: number;
  threshold: number;
}

export interface OpenClawConfig {
  enabled: boolean;
  mode: "local" | "ssh";
  state_dir: string;
  ssh_host: string;
  ssh_user: string;
  ssh_key_path: string;
  ssh_port: number;
  watcher_enabled: boolean;
  watcher_poll_interval: number;
  auto_monitor_enabled: boolean;
  max_auto_sessions: number;
  auto_monitor_agent: string;
}

export interface EmbeddingConfig {
  enabled: boolean;
  model_name: string;
  chunk_size: number;
  chunk_overlap: number;
}

export interface MemoryConfig {
  enabled: boolean;
  /** Legacy combined flag — when true, implicitly enables both layers. */
  inject_enabled: boolean;
  /** Layer 1: inject MEMORY.md once at session start. */
  inject_on_session_start: boolean;
  /** Layer 2: per-turn relevance ranker picks topic files each turn. */
  inject_realtime: boolean;
  model_name: string;
  /** follow_main | frontier | mlx — which stack runs consolidation + ranking. */
  llm_family?: string;
  /** HF repo id used when ``llm_family`` is ``mlx``; blank falls back to global MLX text model. */
  mlx_model?: string;
  min_hours: number;
  min_sessions: number;
  retention_days: number;
  max_memory_files: number;
  max_index_kb: number;
  /** Semantic embedding index sub-feature (Apple Silicon only). */
  embedding?: EmbeddingConfig;
}

export interface MemoryTopic {
  filename: string;
  name: string;
  description: string;
  type: string;
  confidence: string;
  created_at: string | null;
  updated_at: string | null;
  source_sessions: string[];
  size_bytes: number;
  content?: string;
}

export interface OrchestratorConfig {
  /** follow_main | frontier | mlx */
  llm_family: string;
  mlx_model: string;
  /** llm | vlm */
  mlx_model_type: string;
  /** When set, wins over ``llm_family`` (e.g. anthropic / mlx). */
  provider_override: string | null;
  /**
   * Orchestrator system prompt size mode.
   *   ``"auto"`` (default) — lite when orchestrator runs on mlx/exo, full otherwise.
   *   ``"full"``           — force long Claude-tuned prompt regardless of provider.
   *   ``"lite"``           — force short prompt regardless of provider.
   */
  prompt_mode?: "auto" | "full" | "lite";
  /**
   * LangGraph recursion limit (max agent steps per run). Range 1–10000.
   * Defaults to 1000 (deepagents library default).
   */
  recursion_limit?: number;
}

// ---------------------------------------------------------------------------
// EXO cluster
// ---------------------------------------------------------------------------

export interface ExoRemote {
  ssh_alias: string;
  label: string;
  app_data_dir: string;
  enabled: boolean;
  /** Local private key whose pubkey was installed during bootstrap (informational). */
  identity_file?: string;
}

// ---------------------------------------------------------------------------
// EXO Cluster setup wizard (Settings → LLM → Cluster → "Add remote")
// ---------------------------------------------------------------------------

/** Result of a non-interactive ``ssh user@host:port`` probe (no password). */
export interface SshProbeResult {
  tcp_reachable: boolean;
  key_auth_ok: boolean;
  password_auth_available: boolean;
  os_name: string;
  arch: string;
  has_uv: boolean;
  has_exo: boolean;
  hostname_canonical: string;
  error: string;
  hint: string;
}

/** A keypair candidate found under ``~/.ssh/`` (no private key bytes returned). */
export interface LocalKeypair {
  private_path: string;
  public_path: string;
  key_type: string;
  fingerprint: string;
  bits: number;
  comment: string;
}

/** Result of installing a public key on the remote (no password echoed back). */
export interface InstallPubkeyResult {
  ok: boolean;
  fingerprint: string;
  key_type: string;
  bits: number;
  already_present: boolean;
  /** The host actually used for the connection (may differ from the
   *  requested host when the backend auto-swapped to a Bonjour name). */
  host_used?: string;
  /** Set when the backend automatically retried with a Bonjour name
   *  after the original link-local IP was unroutable. */
  host_swapped_from?: string;
}

/** Result of appending a ``Host`` block to ``~/.ssh/config``. */
export interface SshConfigAppendResult {
  config_path: string;
  /** Empty when the file didn't exist yet (no backup needed). */
  backup_path: string;
  appended_block: string;
  replaced: boolean;
  options_used: Record<string, string>;
}

/** Convenience prefill for the wizard's host form. */
export interface ExoSetupLocalUser {
  user: string;
  home: string;
  ssh_dir: string;
  ssh_dir_exists: boolean;
  ssh_dir_perm_ok: boolean;
  platform: string;
}

/** Live Thunderbolt-Bridge link snapshot. ``connected: false`` when no TB cable. */
export interface ExoTbLinkSnapshot {
  connected: boolean;
  interface?: string;
  local_subnets?: string[];
  peer_candidates?: string[];
  reachable_peer?: string | null;
}

export interface SshConfigHost {
  alias: string;
  hostname: string;
  user: string;
  port: number;
  identity_file: string;
  source_file: string;
}

export interface LanSshHost {
  name: string;
  hostname: string;
  port: number;
  addresses: string[];
  thunderbolt_addresses: string[];
  matches_alias: string;
}

export interface ExoConfig {
  enabled: boolean;
  /**
   * Runtime delivery mode:
   *  - "prebuilt" (default): download a notarized prebuilt runtime on demand
   *    (no git/uv/npm/rustup on the user's machine).
   *  - "source": legacy clone + `uv sync` local build (advanced / fallback).
   */
  mode: "prebuilt" | "source";
  /**
   * Custom prebuilt artifact source. Blank = default GitHub releases manifest.
   * Accepts a manifest JSON URL, a direct .tar.gz URL, or a file:// path.
   */
  prebuilt_url: string;
  repo_url: string;
  repo_ref: string;
  api_port: number;
  libp2p_port: number;
  base_url: string;
  model_name: string;
  auto_start: boolean;
  auto_provision: boolean;
  no_terminal_wrap: boolean;
  /**
   * Minimum cluster nodes that must hold a shard of a placed instance.
   * exo's POST /place_instance defaults this to 1 (cheapest single-node
   * placement when the model fits). Set to 2+ to force a pipeline split
   * across multiple macs.
   */
  min_nodes: number;
  /** Max tokens generated per response, sent per request as max_tokens. Bounds runaway generations. */
  max_tokens: number;
  /** Chain-of-thought reasoning, sent per request as enable_thinking (Qwen3/DeepSeek/GLM). Off = fastest. */
  enable_thinking: boolean;
  /** Sharding strategy for multi-node placements: "Pipeline" (best single-request latency) | "Tensor" (throughput). */
  sharding: string;
  /** Instance collective backend: "MlxRing" (universal) | "MlxJaccl" (Thunderbolt 5 / RDMA only). */
  instance_meta: string;
  remotes: ExoRemote[];
}

export interface ExoNodeInfo {
  node_id: string;
  chip: string | null;
  friendly_name: string | null;
  memory_total_gb: number | null;
  memory_free_gb: number | null;
}

export interface ExoStatus {
  reachable: boolean;
  base_url: string;
  master_node_id: string | null;
  peer_count: number;
  rdma_connections: number;
  loaded_models: string[];
  instances: string[];
  runners: string[];
  nodes: ExoNodeInfo[];
  error?: string | null;
}

export interface ExoProvisionState {
  exo_ref: string;
  git_commit: string;
  deps_installed_for_commit: string;
  dashboard_built_for_commit: string;
  last_success_at: string;
  last_error: string;
  exo_repo_dir: string;
  /**
   * Human-readable warning when EXO's pinned MLX differs from Otto's bundled
   * MLX (the "MLX-version trap").  Empty string means preflight passed or
   * hasn't run yet; ``mlx_version_pinned`` and ``mlx_version_bundled``
   * carry the raw version strings for the UI to render side-by-side.
   */
  mlx_version_warning: string;
  mlx_version_pinned: string;
  mlx_version_bundled: string;
}

export interface ExoPrereqs {
  brew: string | null;
  uv: string | null;
  node: string | null;
  npm: string | null;
  git: string | null;
  rustup: string | null;
  cargo: string | null;
  rust_nightly: boolean;
  platform: string;
}

export interface ExoPrebuiltState {
  exo_ref: string;
  sha256: string;
  arch: string;
  source_url: string;
  installed_at: string;
}

export interface ExoInfo {
  platform: string;
  python: string;
  app_data_dir: string;
  exo_root: string;
  exo_repo_dir: string;
  state_file: string;
  pid_file: string;
  log_file: string;
  state: ExoProvisionState;
  prereqs: ExoPrereqs;
  config: ExoConfig;
  running: boolean;
  /**
   * Delivery mode that will actually be used given the host (prebuilt is
   * Apple-Silicon only and silently falls back to source elsewhere).
   */
  mode_effective: "prebuilt" | "source";
  /**
   * Single, mode-aware signal for "is the runtime ready to start?". Use this
   * instead of sniffing source-only fields like ``state.exo_repo_dir``.
   */
  installed: boolean;
  /** Prebuilt runtime metadata; null when running in source mode. */
  prebuilt: ExoPrebuiltState | null;
}

export interface ExoJobPhase {
  name: string;
  status: "pending" | "running" | "done" | "error";
  message: string;
}

export interface ExoJob {
  id: string;
  kind: string;
  target: string;
  status: "pending" | "running" | "done" | "error";
  started_at: number;
  finished_at: number | null;
  error: string | null;
  log_lines: string[];
  result: Record<string, unknown> | null;
  phases: ExoJobPhase[];
}

export interface ExoCatalogModel {
  id: string;
  name: string;
  downloaded: boolean;
  loaded: boolean;
}

export interface ExoModelsResponse {
  reachable: boolean;
  base_url: string;
  models: ExoCatalogModel[];
  error?: string;
}

export interface ExoPreloadResult {
  ok: boolean;
  model: string;
  elapsed_seconds: number;
  first_choice?: string;
  error?: string;
  detail?: string;
  /** Number of pre-existing instances that were deleted before the new placement. */
  replaced?: number;
}

// ---------------------------------------------------------------------------
// EXO catalog (cluster-aware fit scoring) — mirrors backend/exo_catalog.py
// ---------------------------------------------------------------------------

export interface ExoCatalogRow {
  model_id: string;
  family: string;
  base_model: string;
  quant: string;
  weights_gb: number;
  params_b: number;
  n_layers: number;
  hidden_size: number;
  num_kv_heads: number;
  context_length: number;
  capabilities: string[];
  tier: "starter" | "balanced" | "power" | "frontier" | string;
  downloaded: boolean;
  loaded: boolean;
  featured: boolean;
  // Score fields:
  total_gb: number;
  kv_gb: number;
  overhead_gb: number;
  per_node_gb: number;
  bottleneck_gb: number;
  ceiling_gb: number;
  fits: "comfortable" | "tight" | "over" | "unknown";
  /** Smallest min_nodes value (within current cluster) that fits comfortably,
   *  or null if even the whole cluster wouldn't help. */
  min_nodes_required: number | null;
}

export interface ExoCatalogResponse {
  rows: ExoCatalogRow[];
  counts: {
    total: number;
    comfortable: number;
    tight: number;
    over: number;
    downloaded: number;
    loaded: number;
  };
  cluster: {
    reachable: boolean;
    peer_count: number;
    min_nodes: number;
    max_nodes: number;
    nodes: ExoNodeInfo[];
    error?: string | null;
  };
  params: { ctx_len: number; kv_bits: number | null };
}

export type ExoPreloadStage =
  | "placing"
  | "downloading"
  | "loading"
  | "done"
  | "error"
  | "cancelled";

export interface ExoPreloadJob {
  job_id: string;
  model_id: string;
  base_url: string;
  min_nodes: number;
  stage: ExoPreloadStage;
  status: "running" | "done" | "error" | "cancelled";
  message: string;
  started_at: number | null;
  elapsed_seconds: number;
  bytes_done: number;
  bytes_total: number;
  files_done: number;
  files_total: number;
  rate_bps: number;
  eta_seconds: number | null;
  /** Number of cluster nodes currently advancing the download. */
  nodes_active: number;
  /** Pre-existing instances replaced when the job started. */
  replaced: number;
}

export interface ActivityConfig {
  enabled: boolean;
  interval_secs: number;
  retain_days: number;
  exclude_apps: string[];
  idle_threshold_secs: number;
  min_span_secs: number;
  max_span_secs: number;
  context_max_chars: number;
  field_val_max_chars: number;
  browser_text_max_chars: number;
  ax_walk_max_chars: number;
  ax_walk_max_depth: number;
  max_db_mb: number;
}

export interface OmlxConfig {
  enabled: boolean;
  api_port: number;
  base_url: string;
  model_name: string;
  auto_start: boolean;
  brew_tap: string;
  brew_tap_url: string;
  brew_formula: string;
  cli_path: string;
  /** Directories oMLX scans for local models. Tilde (~) is expanded by the backend. */
  model_dirs: string[];
  max_context_window: number;
  /** Chain-of-thought reasoning, sent per request as chat_template_kwargs.enable_thinking. */
  thinking_enabled: boolean;
  /** Max tokens generated per response, sent per request as max_tokens. Bounds runaway generations. */
  max_tokens: number;
}

export interface OmlxDetection {
  cli_path: string | null;
  app_bundle: string | null;
  homebrew: boolean;
  brew_service_state: string | null;
  cli_version: string | null;
  installed: boolean;
}

export interface OmlxInfo {
  config: OmlxConfig;
  detection: OmlxDetection;
  spawn_pid: number | null;
  spawn_log_path: string;
}

export interface OmlxStatus {
  base_url: string;
  reachable: boolean;
  /** All ids registered with oMLX (for ID resolution). */
  models: { id: string }[];
  /** Only the ids actually resident in GPU RAM (from admin API). */
  loaded_models: { id: string }[];
  error?: string | null;
}

export interface OmlxJob {
  id: string;
  kind: "install" | "upgrade" | "uninstall" | "start" | "stop" | "load";
  status: "pending" | "running" | "done" | "error";
  started_at: number;
  finished_at: number | null;
  error: string | null;
  log_lines: string[];
  result: Record<string, unknown> | null;
}

export interface NodeJob {
  id: string;
  kind: string;
  status: "pending" | "running" | "done" | "error";
  started_at: number;
  finished_at: number | null;
  error: string | null;
  log_lines: string[];
  result: Record<string, unknown> | null;
}

export interface OmlxVersionInfo {
  installed_version: string | null;
  latest_version: string | null;
  upgrade_available: boolean;
  homebrew: boolean;
}

export interface OmlxCacheStats {
  reachable: boolean;
  cache_efficiency_pct: number;
  total_requests: number;
  total_tokens_served: number;
  total_cached_tokens: number;
  total_prompt_tokens: number;
  avg_prefill_tps: number;
  avg_generation_tps: number;
  uptime_seconds: number;
  disk_bytes: number;
  disk_gb: number;
  cache_dir: string;
}

export interface OmlxCacheSettings {
  cache_enabled: boolean;
  hot_cache_only: boolean;
  hot_cache_max_size: string;
  ssd_cache_dir: string;
  ssd_cache_max_size: string;
  initial_cache_blocks: number;
  max_context_window: number;
  /** Continuous-batching concurrency cap (max requests decoded in parallel). */
  max_concurrent_requests: number;
}

export interface OmlxLocalModel {
  repo_id: string;
  size_gb: number;
  is_mlx: boolean;
}

export interface OmlxModelCatalogRow {
  repo_id: string;
  display_name: string;
  blurb: string;
  weights_gb: number;
  params_b: number;
  quant: string;
  role: string[];
  capability_tags: string[];
  tier: string;
  featured: boolean;
  downloads: number;
  fits: "comfortable" | "tight" | "over" | "unknown";
  total_gb: number;
  headroom_gb: number;
  disk_ok: boolean;
  already_cached: boolean;
}

export interface PrivacyConfig {
  enabled: boolean;
  local_only_providers: string[];
  allowed_hosts: string[];
  allow_loopback: boolean;
  allow_mdns: boolean;
  pf_anchor: string;
  engaged_at: string;
  audit_token: string;
}

export interface PrivacyStatus extends PrivacyConfig {
  engaged: boolean;
  pf?: {
    available: boolean;
    anchor?: string;
    rule_count?: number;
    has_block_rule?: boolean;
    stdout_excerpt?: string;
    exit_code?: number;
    reason?: string;
  };
}

export interface PrivacyAuditEntry {
  ts: string;
  event: string;
  audit_token: string;
  engaged: boolean;
  provider?: string;
  allowed?: string[];
  rotated?: boolean;
  was_engaged?: boolean;
}

// ---------------------------------------------------------------------------
// Ambient assistant
// ---------------------------------------------------------------------------

export interface AmbientConfig {
  enabled: boolean;
  /** follow_main | frontier | mlx | exo */
  llm_family: string;
  /** HF repo id when llm_family === "mlx" */
  mlx_model: string;
  /** Cloud model name when llm_family === "frontier"; blank = auto-Haiku */
  model_name: string;
  interval_mins: number;
  idle_only: boolean;
  react_to_session_end: boolean;
  use_memory: boolean;
  use_sessions: boolean;
  use_activity: boolean;
  use_history: boolean;
  /** Hours back that sessions, activity, and history gatherers look. Default 24. */
  lookback_hours: number;
  min_confidence: number;
  max_hints_per_day: number;
  cooldown_hours: number;
  quiet_hours_start: number;
  quiet_hours_end: number;
  allow_auto_run: boolean;
}

export type AmbientHintStatus =
  | "pending"
  | "shown"
  | "accepted"
  | "dismissed"
  | "snoozed"
  | "expired";

export type AmbientHintKind = "task" | "automation" | "schedule" | "trigger";

export interface AmbientHint {
  id: string;
  title: string;
  rationale: string;
  proposed_prompt: string;
  suggested_agent: string | null;
  kind: AmbientHintKind;
  schedule_cron: string | null;
  confidence: number;
  sources: string[];
  topic_hash: string;
  status: AmbientHintStatus;
  session_id: string | null;
  created_at: number;
  shown_at: number | null;
  acted_at: number | null;
  snoozed_until: number | null;
  /** "ambient" (default) or "evaluation" for eval-triggered suggestions. */
  origin?: string;
  /** For eval suggestions: "manual" | "schedule" | "trigger". */
  target_kind?: string | null;
  /** Schedule or trigger id when target_kind is schedule/trigger. */
  target_id?: string | null;
}

export interface AppSettings {
  llm: LLMConfig;
  orchestrator: OrchestratorConfig;
  mcp_servers: MCPServerConfig[];
  observability: ObservabilityConfig;
  claude_hook: ClaudeHookConfig;
  evaluation: EvaluationConfig;
  openclaw: OpenClawConfig;
  memory: MemoryConfig;
  exo: ExoConfig;
  omlx: OmlxConfig;
  activity: ActivityConfig;
  privacy: PrivacyConfig;
  auto_approve_commands: boolean;
  ambient_suggest_recurrence: boolean;
  ambient: AmbientConfig;
  voice: VoiceConfig;
}

// ---------------------------------------------------------------------------
// Voice subsystem
// ---------------------------------------------------------------------------

export type VoiceState =
  | "idle"
  | "listening"
  | "capturing"
  | "transcribing";

export type VoiceWSEventType =
  | "state"
  | "transcript"
  | "partial"
  | "wake"
  | "error";

export interface VoiceWSEvent {
  type: VoiceWSEventType;
  state?: VoiceState;
  text?: string;
  message?: string;
}

export interface VoiceConfig {
  enabled: boolean;
  activation_mode: string;
  ptt_hotkey: string;
  stt_enabled: boolean;
  stt_model: string;
  stt_language: string;
  wake_enabled: boolean;
  wake_model: string;
  vad_silence_secs: number;
  mic_device: string;
  loopback_enabled: boolean;
  loopback_vad_silence_secs: number;
  loopback_max_segment_secs: number;
  loopback_live_partials: boolean;
  loopback_partial_interval_secs: number;
  loopback_auto_send_silence_secs: number;
}

// ---------------------------------------------------------------------------
// System-audio (loopback) transcription
// ---------------------------------------------------------------------------

export type TranscribeState = "idle" | "recording";

export type TranscribeSource = "system" | "mic";

export type TranscribeWSEventType =
  | "state"
  | "partial"
  | "segment"
  | "level"
  | "model"
  | "error";

/** Readiness of the on-device speech (Whisper) model. */
export type TranscribeModelStatus = "ready" | "downloading" | "error";

export interface TranscribeWSEvent {
  type: TranscribeWSEventType;
  state?: TranscribeState;
  text?: string;
  message?: string;
  ts?: number;
  rms?: number;
  source?: TranscribeSource;
  /** Stable, run-scoped event id used to dedupe across multiple sockets. */
  eid?: string;
  /** For "model" events: model readiness/download status. */
  status?: TranscribeModelStatus;
  /** For "model" events: repo id of the speech model. */
  model?: string;
  /** For "model" events: download progress 0..1 (absent = indeterminate). */
  progress?: number;
}

export interface TranscriptSegment {
  id: string;
  text: string;
  ts: number;
  source: TranscribeSource;
}

export interface LoopbackStatus {
  state: TranscribeState;
  supported: boolean;
  helper_available: boolean;
  mic_available: boolean;
  config: {
    loopback_enabled: boolean;
    loopback_vad_silence_secs: number;
    loopback_max_segment_secs: number;
    loopback_live_partials: boolean;
    loopback_partial_interval_secs: number;
    loopback_auto_send_silence_secs: number;
  };
}

// ── Screen capture (Transcribe screenshots) ─────────────────────────────
export interface CaptureWindow {
  window_id: number;
  app: string;
  title: string;
  /** Small base64 PNG preview (best-effort; requires Screen Recording). */
  thumb_b64?: string;
}

export interface CapturePermission {
  supported: boolean;
  /** true / false / null (unknown). */
  granted: boolean | null;
  can_prompt: boolean;
}

/** Result of POST /api/capture/screen. */
export interface CaptureResult {
  image_b64?: string;
  mime_type?: string;
  width?: number;
  height?: number;
  hash?: string;
  /** Screen was visually unchanged vs. last_hash — no image returned. */
  unchanged?: boolean;
  /** Screen Recording permission missing. */
  needs_permission?: boolean;
  /** Followed window no longer exists. */
  window_gone?: boolean;
  /** Platform / frameworks unavailable. */
  unsupported?: boolean;
  error?: string;
}

/** The window the user chose to "follow" for transcript-anchored auto-capture. */
export interface FollowedWindow {
  window_id: number;
  app: string;
  title: string;
}

/** A captured screenshot shown inline in the transcript feed. */
export interface Shot {
  id: string;
  /** Full data URL (data:image/png;base64,...) for display + upload. */
  dataUrl: string;
  label: string;
  ts: number;
  /** Discriminator so the feed can distinguish shots from transcript segments. */
  kind: "shot";
}

/** Image attachment carried over the ask-Otto bus. */
export interface AskImage {
  name: string;
  dataUrl: string;
}

export interface VoiceCatalogRow {
  repo_id: string;
  kind: "stt" | "wake";
  display_name: string;
  blurb: string;
  weights_gb: number;
  format: "mlx" | "onnx";
  language: string;
  latency_class: "realtime" | "near-realtime" | "batch";
  wake_phrases: string[];
  featured: boolean;
  downloads: number;
  last_modified: string;
  // Fit scoring (from backend)
  total_gb: number;
  fits: "comfortable" | "tight" | "over" | "unknown";
  disk_ok: boolean;
  already_cached: boolean;
  cache_complete: boolean;
}

export interface VoiceStatus {
  state: VoiceState;
  config: VoiceConfig;
  input_devices: Array<{ index: number; name: string; channels: number; default: boolean }>;
}
