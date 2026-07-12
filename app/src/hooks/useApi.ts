import type {
  AgentSpec,
  AppSettings,
  RunsResponse,
  RunStats,
  RunEvaluation,
  SessionTimeline,
  ExoInfo,
  PrivacyAuditEntry,
  PrivacyStatus,
  ExoJob,
  NodeJob,
  OmlxCacheSettings,
  OmlxCacheStats,
  OmlxInfo,
  OmlxJob,
  OmlxLocalModel,
  OmlxModelCatalogRow,
  OmlxStatus,
  OmlxVersionInfo,
  ExoCatalogResponse,
  ExoModelsResponse,
  ExoPreloadJob,
  ExoPreloadResult,
  ExoRemote,
  ExoSetupLocalUser,
  ExoStatus,
  ExoTbLinkSnapshot,
  InstallPubkeyResult,
  LanSshHost,
  LocalKeypair,
  MCPAuthStatus,
  MCPServerStatus,
  MlxCapabilities,
  MlxLiveStats,
  MlxCatalogResponse,
  MlxDownloadJob,
  ScheduleAttachment,
  ScheduleRunsResponse,
  ScheduleRunStats,
  ScheduleSpec,
  ScheduleStatusPoll,
  SessionInfo,
  SkillSpec,
  TriggerRunsResponse,
  TriggerSpec,
  TriggerStatusPoll,
  SshConfigAppendResult,
  SshConfigHost,
  SshProbeResult,
} from "../types";
import { API_BASE } from "../config/apiBase";

export interface MemoryHitsResponse {
  total_injections: number;
  unique_sessions: number;
  cache_hit_rate: number;
  top_topics: { topic: string; count: number }[];
  recent: { ts: string; session_id: string; topics: string[]; query: string }[];
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  // Health
  health: () => request<{ status: string }>("/api/health"),

  // Settings
  getSettings: () => request<AppSettings>("/api/settings"),
  updateSettings: (data: Record<string, unknown>) =>
    request<{ status: string }>("/api/settings", { method: "PUT", body: JSON.stringify(data) }),
  isFirstRun: () =>
    request<{
      first_run: boolean;
      completed: boolean;
      dismissed: boolean;
      current_step: string;
      completed_steps: string[];
    }>("/api/settings/first-run"),

  // First-run setup wizard
  setupState: () =>
    request<{
      completed: boolean;
      dismissed: boolean;
      current_step: string;
      completed_steps: string[];
      first_run: boolean;
    }>("/api/setup/state"),
  setupMarkStep: (step: string, completed = true) =>
    request<{ current_step: string; completed_steps: string[] }>("/api/setup/step", {
      method: "POST",
      body: JSON.stringify({ step, completed }),
    }),
  setupChat: (data: {
    step: string;
    user_message: string;
    extracted?: string | null;
    context?: Record<string, unknown>;
    model_ready?: boolean;
  }) =>
    request<{ reply: string; needs_clarification: boolean; setup_model_id: string }>(
      "/api/setup/chat",
      { method: "POST", body: JSON.stringify(data) },
    ),
  setupComplete: () =>
    request<{ ok: boolean }>("/api/setup/complete", { method: "POST" }),
  setupSkip: () =>
    request<{ ok: boolean }>("/api/setup/skip", { method: "POST" }),
  setupReset: () =>
    request<{ ok: boolean }>("/api/setup/reset", { method: "POST" }),
  accessibilityPermission: () =>
    request<{
      platform: string;
      supported: boolean;
      granted: boolean;
      can_prompt: boolean;
      error?: string;
    }>("/api/setup/permissions/accessibility"),
  promptAccessibilityPermission: () =>
    request<{ ok: boolean; granted?: boolean; error?: string }>(
      "/api/setup/permissions/prompt",
      { method: "POST", body: JSON.stringify({ kind: "accessibility" }) },
    ),
  testConnection: (data: Record<string, unknown>) =>
    request<{ success: boolean; message: string; models?: { id: string; name: string }[] }>("/api/settings/test-connection", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  listModels: (data: Record<string, unknown>) =>
    request<{ models: { id: string; name: string }[]; error?: string }>("/api/settings/list-models", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  mlxHubDefault: () =>
    request<{ path: string; cache_root: string; default_suffix: string; default_relative?: string }>(
      "/api/mlx/hub-default",
    ),
  mlxLocalModels: (queryString = "") =>
    request<{
      hub_cache: string;
      models: { repo_id: string; name: string; size_mb: number }[];
      error: string | null;
    }>(`/api/mlx/local-models${queryString}`),
  mlxDownload: (data: Record<string, unknown>) =>
    request<{ job_id: string; repo_id: string; hub_cache: string }>("/api/mlx/download", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  mlxDownloadStatus: (jobId: string) =>
    request<MlxDownloadJob>(
      `/api/mlx/download/${encodeURIComponent(jobId)}`,
    ),
  mlxDownloadList: () =>
    request<{ jobs: MlxDownloadJob[] }>("/api/mlx/downloads"),
  mlxDownloadCancel: (jobId: string) =>
    request<{ status: string; job_id: string }>(
      `/api/mlx/download/${encodeURIComponent(jobId)}/cancel`,
      { method: "POST" },
    ),

  // Setup Hub: capabilities probe + scored MLX catalog.
  mlxCapabilities: () =>
    request<MlxCapabilities>("/api/mlx/capabilities"),
  getMlxLiveStats: () =>
    request<MlxLiveStats>("/api/mlx/live-stats"),
  mlxCatalog: (params?: { ctx_len?: number; kv_bits?: number | null; refresh?: boolean }) => {
    const sp = new URLSearchParams();
    if (params?.ctx_len) sp.set("ctx_len", String(params.ctx_len));
    if (params?.kv_bits != null) sp.set("kv_bits", String(params.kv_bits));
    if (params?.refresh) sp.set("refresh", "true");
    const q = sp.toString();
    return request<MlxCatalogResponse>(`/api/mlx/catalog${q ? `?${q}` : ""}`);
  },
  mlxTurboSsdCacheInfo: () =>
    request<{
      root: string;
      exists: boolean;
      entries: number;
      total_bytes: number;
      total_gb: number;
      max_gb: number;
      models: { model_slug: string; entries: number; size_bytes: number; size_gb: number }[];
    }>("/api/mlx/turbo/ssd-cache"),
  mlxTurboSsdCacheClear: (modelSlug?: string) =>
    request<{ root: string; scope: string; removed_files: number }>(
      `/api/mlx/turbo/ssd-cache${modelSlug ? `?model=${encodeURIComponent(modelSlug)}` : ""}`,
      { method: "DELETE" },
    ),

  testOpenClawConnection: () =>
    request<{ success: boolean; message: string }>("/api/settings/openclaw/test", { method: "POST" }),

  installClaudeHooks: () =>
    request<{ status: string; message: string; path?: string }>("/api/settings/claude-hooks/install", { method: "POST" }),

  claudeHookStatus: () =>
    request<{
      enabled: boolean;
      active_sessions: number;
      auto_monitor: { enabled: boolean; active: Record<string, string> };
    }>("/hooks/claude/status"),

  openclawStatus: () =>
    request<{
      enabled: boolean;
      running: boolean;
      poll_interval?: number;
      tracked_agents?: number;
      tracked_sessions?: number;
      buffered_events?: number;
      auto_monitor?: { active: Record<string, string>; count: number };
      error?: string;
    }>("/hooks/openclaw/status"),

  // MCP Servers
  listMCPServers: () => request<MCPServerStatus[]>("/api/mcp-servers"),
  addMCPServer: (data: Record<string, unknown>) =>
    request<{ status: string; id: string }>("/api/mcp-servers", { method: "POST", body: JSON.stringify(data) }),
  updateMCPServer: (id: string, data: Record<string, unknown>) =>
    request<{ status: string; id: string }>(`/api/mcp-servers/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  updateExcludedTools: (id: string, excludedTools: string[]) =>
    request<{ status: string; excluded_tools: string[] }>(`/api/mcp-servers/${id}/excluded-tools`, {
      method: "PUT",
      body: JSON.stringify({ excluded_tools: excludedTools }),
    }),
  removeMCPServer: (id: string) =>
    request<{ status: string }>(`/api/mcp-servers/${id}`, { method: "DELETE" }),
  connectMCPServer: (id: string) =>
    request<MCPServerStatus>(`/api/mcp-servers/${id}/connect`, { method: "POST" }),
  testMCPServer: (id: string) =>
    request<{ success: boolean; message: string }>(`/api/mcp-servers/${id}/test`, {
      method: "POST",
    }),
  startMCPProcess: (id: string) =>
    request<Record<string, unknown>>(`/api/mcp-servers/${id}/start`, { method: "POST" }),
  stopMCPProcess: (id: string) =>
    request<Record<string, unknown>>(`/api/mcp-servers/${id}/stop`, { method: "POST" }),

  // Interactive MCP auth flows.  ``mcpAuthLogin`` runs the server's
  // configured provider (OAuth device, OAuth auth-code, or
  // browser-capture) inside the backend and persists the resulting
  // token bundle in the OS keychain.  Frontend never sees the token —
  // only the boolean / expiry status from ``mcpAuthStatus``.
  mcpAuthStatus: (id: string) =>
    request<MCPAuthStatus>(`/api/mcp-servers/${encodeURIComponent(id)}/auth/status`),
  mcpAuthLogin: (id: string) =>
    request<{
      status: "ok" | "error";
      auth?: MCPAuthStatus;
      auth_kind?: string;
      reason?: string;
      message?: string;
    }>(`/api/mcp-servers/${encodeURIComponent(id)}/auth/login`, {
      method: "POST",
    }),
  mcpAuthLogout: (id: string) =>
    request<{ status: "logged_out" | "no_bundle"; auth: MCPAuthStatus }>(
      `/api/mcp-servers/${encodeURIComponent(id)}/auth/logout`,
      { method: "POST" },
    ),
  exportMCPServers: () =>
    request<{ mcpServers: Record<string, unknown> }>("/api/mcp-servers/export"),
  importMCPServers: (data: Record<string, unknown>) =>
    request<{ added: string[]; skipped: string[] }>("/api/mcp-servers/import", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  saveMCPServersJson: (data: Record<string, unknown>) =>
    request<{ status?: string; error?: string; details?: string[]; count?: number }>("/api/mcp-config", {
      method: "PUT",
      body: JSON.stringify(data),
    }),

  // Credential vault — backed by macOS Keychain on Darwin, Secret
  // Service on Linux, Credential Manager on Windows.  Values never
  // come back from the API; only names and a boolean availability flag.
  // Setting a value goes to the OS keychain; the backend then
  // hydrates env vars at MCP subprocess spawn time.
  vaultHealth: () =>
    request<{ available: boolean; backend: string | null; error?: string }>("/api/vault/health"),
  vaultListServers: () =>
    request<{ servers: string[] }>("/api/vault/secrets"),
  vaultListNames: (serverId: string) =>
    request<{ server_id: string; names: string[] }>(
      `/api/vault/secrets/${encodeURIComponent(serverId)}`,
    ),
  vaultSetSecret: (serverId: string, name: string, value: string) =>
    request<{ status: string; server_id: string; name: string }>(
      `/api/vault/secrets/${encodeURIComponent(serverId)}/${encodeURIComponent(name)}`,
      { method: "POST", body: JSON.stringify({ value }) },
    ),
  vaultDeleteSecret: (serverId: string, name: string) =>
    request<{ status: string; server_id: string; name: string }>(
      `/api/vault/secrets/${encodeURIComponent(serverId)}/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),
  vaultDeleteAllForServer: (serverId: string) =>
    request<{ status: string; server_id: string; removed: number }>(
      `/api/vault/secrets/${encodeURIComponent(serverId)}`,
      { method: "DELETE" },
    ),

  // Agents
  listAgents: () => request<AgentSpec[]>("/api/agents"),
  getAgent: (name: string) => request<AgentSpec>(`/api/agents/${encodeURIComponent(name)}`),
  createAgent: (data: Record<string, unknown>) =>
    request<AgentSpec>("/api/agents", { method: "POST", body: JSON.stringify(data) }),
  updateAgent: (name: string, data: Record<string, unknown>) =>
    request<AgentSpec>(`/api/agents/${encodeURIComponent(name)}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteAgent: (name: string) =>
    request<{ deleted: boolean }>(`/api/agents/${encodeURIComponent(name)}`, { method: "DELETE" }),
  generateAgent: (description: string) =>
    request<Record<string, unknown>>("/api/agents/generate", {
      method: "POST",
      body: JSON.stringify({ user_description: description }),
    }),
  exportAgents: () =>
    request<{ agents: Record<string, unknown> }>("/api/agents-config"),
  saveAgentsJson: (data: Record<string, unknown>) =>
    request<{ status?: string; error?: string; details?: string[]; count?: number }>("/api/agents-config", {
      method: "PUT",
      body: JSON.stringify(data),
    }),

  // Skills
  listSkills: () => request<SkillSpec[]>("/api/skills"),
  getSkill: (name: string) => request<SkillSpec>(`/api/skills/${encodeURIComponent(name)}`),
  createSkill: (data: Record<string, unknown>) =>
    request<SkillSpec>("/api/skills", { method: "POST", body: JSON.stringify(data) }),
  updateSkill: (name: string, data: Record<string, unknown>) =>
    request<SkillSpec>(`/api/skills/${encodeURIComponent(name)}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteSkill: (name: string) =>
    request<{ deleted: boolean }>(`/api/skills/${encodeURIComponent(name)}`, { method: "DELETE" }),
  generateSkill: (description: string) =>
    request<{ content: string }>("/api/skills/generate", {
      method: "POST",
      body: JSON.stringify({ user_description: description }),
    }),
  exportSkills: () =>
    request<{ skills: Record<string, unknown> }>("/api/skills-config"),
  saveSkillsJson: (data: Record<string, unknown>) =>
    request<{ status?: string; error?: string; details?: string[]; count?: number }>("/api/skills-config", {
      method: "PUT",
      body: JSON.stringify(data),
    }),

  // Schedules
  listSchedules: () => request<ScheduleSpec[]>("/api/schedules"),
  createSchedule: (data: Record<string, unknown>) =>
    request<ScheduleSpec>("/api/schedules", { method: "POST", body: JSON.stringify(data) }),
  updateSchedule: (id: string, data: Record<string, unknown>) =>
    request<ScheduleSpec>(`/api/schedules/${encodeURIComponent(id)}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteSchedule: (id: string) =>
    request<{ deleted: boolean }>(`/api/schedules/${encodeURIComponent(id)}`, { method: "DELETE" }),
  toggleSchedule: (id: string) =>
    request<ScheduleSpec>(`/api/schedules/${encodeURIComponent(id)}/toggle`, { method: "POST" }),
  runScheduleNow: (id: string) =>
    request<{ status: string }>(`/api/schedules/${encodeURIComponent(id)}/run-now`, { method: "POST" }),
  stopScheduleRun: (id: string) =>
    request<{ status: string }>(`/api/schedules/${encodeURIComponent(id)}/stop`, { method: "POST" }),
  getScheduleRuns: (id: string, params?: { limit?: number; offset?: number; after?: string; before?: string; status?: string; search?: string }) => {
    const qs = new URLSearchParams();
    if (params?.limit != null) qs.set("limit", String(params.limit));
    if (params?.offset != null) qs.set("offset", String(params.offset));
    if (params?.after) qs.set("after", params.after);
    if (params?.before) qs.set("before", params.before);
    if (params?.status) qs.set("status", params.status);
    if (params?.search) qs.set("search", params.search);
    const query = qs.toString();
    return request<ScheduleRunsResponse>(`/api/schedules/${encodeURIComponent(id)}/runs${query ? `?${query}` : ""}`);
  },
  getScheduleRunStats: (id: string, params?: { after?: string; before?: string; status?: string; search?: string }) => {
    const qs = new URLSearchParams();
    if (params?.after) qs.set("after", params.after);
    if (params?.before) qs.set("before", params.before);
    if (params?.status) qs.set("status", params.status);
    if (params?.search) qs.set("search", params.search);
    const query = qs.toString();
    return request<ScheduleRunStats>(`/api/schedules/${encodeURIComponent(id)}/runs/stats${query ? `?${query}` : ""}`);
  },
  openScheduleFolder: (id: string) =>
    request<{ status: string; path: string }>(`/api/schedules/${encodeURIComponent(id)}/open-folder`, { method: "POST" }),
  openScheduleRunFolder: (id: string, runId: string) =>
    request<{ status: string; path: string }>(`/api/schedules/${encodeURIComponent(id)}/runs/${encodeURIComponent(runId)}/open-folder`, { method: "POST" }),
  getScheduleStatus: () =>
    request<ScheduleStatusPoll>("/api/schedules/status/poll"),
  listScheduleAttachments: (id: string) =>
    request<ScheduleAttachment[]>(`/api/schedules/${encodeURIComponent(id)}/attachments`),
  uploadScheduleAttachment: async (
    id: string,
    filename: string,
    file: File,
  ): Promise<{ status: string; path: string; size: number }> => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(
      `${API_BASE}/api/schedules/${encodeURIComponent(id)}/attachments/${encodeURIComponent(filename)}`,
      { method: "POST", body: form },
    );
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  },
  deleteScheduleAttachment: (id: string, path: string) =>
    request<{ status: string; path: string }>(
      `/api/schedules/${encodeURIComponent(id)}/attachments/${path.split("/").map(encodeURIComponent).join("/")}`,
      { method: "DELETE" },
    ),

  // Custom Triggers
  listTriggers: () => request<TriggerSpec[]>("/api/triggers"),
  createTrigger: (data: Record<string, unknown>) =>
    request<TriggerSpec>("/api/triggers", { method: "POST", body: JSON.stringify(data) }),
  updateTrigger: (id: string, data: Record<string, unknown>) =>
    request<TriggerSpec>(`/api/triggers/${encodeURIComponent(id)}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),
  deleteTrigger: (id: string) =>
    request<{ deleted: boolean }>(`/api/triggers/${encodeURIComponent(id)}`, { method: "DELETE" }),
  toggleTrigger: (id: string) =>
    request<TriggerSpec>(`/api/triggers/${encodeURIComponent(id)}/toggle`, { method: "POST" }),
  runTriggerNow: (id: string) =>
    request<{ status: string }>(`/api/triggers/${encodeURIComponent(id)}/run-now`, { method: "POST" }),
  stopTriggerRun: (id: string) =>
    request<{ status: string }>(`/api/triggers/${encodeURIComponent(id)}/stop`, { method: "POST" }),
  getTriggerRuns: (id: string, params?: { limit?: number; offset?: number; after?: string; before?: string; status?: string }) => {
    const qs = new URLSearchParams();
    if (params?.limit != null) qs.set("limit", String(params.limit));
    if (params?.offset != null) qs.set("offset", String(params.offset));
    if (params?.after) qs.set("after", params.after);
    if (params?.before) qs.set("before", params.before);
    if (params?.status) qs.set("status", params.status);
    const query = qs.toString();
    return request<TriggerRunsResponse>(`/api/triggers/${encodeURIComponent(id)}/runs${query ? `?${query}` : ""}`);
  },
  openTriggerFolder: (id: string) =>
    request<{ status: string; path: string }>(
      `/api/triggers/${encodeURIComponent(id)}/open-folder`, { method: "POST" },
    ),
  openTriggerRunFolder: (id: string, runId: string) =>
    request<{ status: string; path: string }>(
      `/api/triggers/${encodeURIComponent(id)}/runs/${encodeURIComponent(runId)}/open-folder`, { method: "POST" },
    ),
  getTriggerStatus: () =>
    request<TriggerStatusPoll>("/api/triggers/status/poll"),

  // Memory
  getMemoryStatus: () =>
    request<{ state: string; started_at: string | null; finished_at: string | null; error: string | null; transcripts_processed: number }>("/api/memory/status"),
  triggerMemoryConsolidation: () =>
    request<{ status: string; transcripts?: number; reason?: string; error?: string }>("/api/memory/run", { method: "POST" }),
  cancelMemoryConsolidation: () =>
    request<{ status: string }>("/api/memory/cancel", { method: "POST" }),
  getMemoryStats: () =>
    request<{ total_transcripts: number; pending_transcripts: number; memory_files: number; last_consolidated_at: number | null; retention_days: number }>("/api/memory/stats"),
  getMemoryHits: () =>
    request<MemoryHitsResponse>("/api/memory/hits"),
  listMemoryTopics: () =>
    request<{ topics: import("../types").MemoryTopic[] }>("/api/memory/topics"),
  getMemoryTopic: (filename: string) =>
    request<import("../types").MemoryTopic>(`/api/memory/topics/${encodeURIComponent(filename)}`),
  updateMemoryTopic: (filename: string, content: string) =>
    request<{ status: string; updated_at: string }>(
      `/api/memory/topics/${encodeURIComponent(filename)}`,
      { method: "PUT", body: JSON.stringify({ content }) },
    ),
  deleteMemoryTopic: (filename: string) =>
    request<{ status: string }>(`/api/memory/topics/${encodeURIComponent(filename)}`, { method: "DELETE" }),
  addMemoryCorrection: (filename: string, text: string) =>
    request<{ status: string; correction: string }>(
      `/api/memory/topics/${encodeURIComponent(filename)}/correction`,
      { method: "POST", body: JSON.stringify({ text }) },
    ),

  // Embedding index
  getEmbeddingModelStatus: () =>
    request<{ installed: boolean; model_name: string; downloading: boolean; bytes_downloaded: number; total_bytes: number; error: string | null }>("/api/embeddings/model-status"),
  startModelDownload: () =>
    request<{ status: string; model_name: string }>("/api/embeddings/model-download", { method: "POST" }),
  getEmbeddingStatus: () =>
    request<{ enabled: boolean; total_chunks: number; sources: { source_path: string; source_type: string; chunk_count: number; indexed_at: number }[]; error?: string }>("/api/embeddings/status"),
  indexPath: (path: string) =>
    request<{ status: string; path: string; type: string }>("/api/embeddings/index", {
      method: "POST", body: JSON.stringify({ path }),
    }),
  removeEmbeddingSource: (path: string) =>
    request<{ status: string; chunks_removed: number }>("/api/embeddings/source", {
      method: "DELETE", body: JSON.stringify({ path }),
    }),
  reindexMemory: () =>
    request<{ status: string }>("/api/embeddings/reindex/memory", { method: "POST" }),

  // EXO cluster lifecycle
  exoInfo: () => request<ExoInfo>("/api/exo"),
  exoStatus: () => request<ExoStatus>("/api/exo/status"),
  exoLog: (lines = 200) =>
    request<{ lines: string[] }>(`/api/exo/log?lines=${lines}`),
  exoUp: (force = false, force_mismatch = false) =>
    request<ExoJob & { job_id: string }>(
      "/api/exo/up",
      { method: "POST", body: JSON.stringify({ force, force_mismatch }) },
    ),
  exoDown: () =>
    request<{ stopped: boolean; running: boolean }>("/api/exo/down", { method: "POST" }),
  exoProvision: (force = false, force_mismatch = false) =>
    request<ExoJob>("/api/exo/provision", {
      method: "POST",
      body: JSON.stringify({ force, force_mismatch }),
    }),
  exoSmoke: () => request<ExoJob>("/api/exo/smoke", { method: "POST" }),
  exoReleaseCheck: () =>
    request<{
      ok: boolean;
      current_ref: string;
      latest_tag?: string;
      newer_available?: boolean;
      html_url?: string;
      published_at?: string;
      error?: string;
    }>("/api/exo/release/check"),
  listExoJobs: () => request<{ jobs: ExoJob[] }>("/api/exo/jobs"),
  getExoJob: (id: string) =>
    request<ExoJob>(`/api/exo/jobs/${encodeURIComponent(id)}`),

  // oMLX local server lifecycle
  omlxInfo: () => request<OmlxInfo>("/api/omlx"),
  omlxStatus: () => request<OmlxStatus>("/api/omlx/status"),
  omlxLog: (lines = 200) =>
    request<{ lines: string[] }>(`/api/omlx/log?lines=${lines}`),
  omlxVersionInfo: () => request<OmlxVersionInfo>("/api/omlx/version"),
  omlxUpgrade: () =>
    request<OmlxJob & { job_id: string }>("/api/omlx/upgrade", { method: "POST" }),
  omlxInstall: () =>
    request<OmlxJob & { job_id: string }>("/api/omlx/install", { method: "POST" }),
  omlxUninstall: () =>
    request<OmlxJob & { job_id: string }>("/api/omlx/uninstall", { method: "POST" }),
  omlxStart: () =>
    request<OmlxJob & { job_id: string }>("/api/omlx/start", { method: "POST" }),
  omlxStop: () =>
    request<OmlxJob & { job_id: string }>("/api/omlx/stop", { method: "POST" }),
  listOmlxJobs: () => request<{ jobs: OmlxJob[] }>("/api/omlx/jobs"),
  getOmlxJob: (id: string) =>
    request<OmlxJob>(`/api/omlx/jobs/${encodeURIComponent(id)}`),
  nodeStatus: () =>
    request<{
      present: boolean;
      system_node: string | null;
      app_data_bin: string;
      app_data_node_exists: boolean;
      pinned_version: string;
    }>("/api/node/status"),
  nodeInstall: (force = false) =>
    request<NodeJob & { job_id: string }>(
      `/api/node/install${force ? "?force=true" : ""}`,
      { method: "POST" },
    ),
  listNodeJobs: () => request<{ jobs: NodeJob[] }>("/api/node/jobs"),
  getNodeJob: (id: string) =>
    request<NodeJob>(`/api/node/jobs/${encodeURIComponent(id)}`),
  omlxLocalModels: () =>
    request<{ models: OmlxLocalModel[]; hub_cache_dir: string; error: string | null }>(
      "/api/omlx/models/local",
    ),
  omlxModelCatalog: (opts?: { ctx_len?: number; refresh?: boolean }) => {
    const params = new URLSearchParams();
    if (opts?.ctx_len) params.set("ctx_len", String(opts.ctx_len));
    if (opts?.refresh) params.set("refresh", "true");
    const qs = params.toString();
    return request<{ capabilities: Record<string, unknown>; models: OmlxModelCatalogRow[] }>(
      `/api/omlx/models/catalog${qs ? `?${qs}` : ""}`,
    );
  },
  omlxSearchModels: (q: string, opts?: { ctx_len?: number; limit?: number }) => {
    const params = new URLSearchParams({ q });
    if (opts?.ctx_len) params.set("ctx_len", String(opts.ctx_len));
    if (opts?.limit) params.set("limit", String(opts.limit));
    return request<{ models: OmlxModelCatalogRow[]; query: string }>(
      `/api/omlx/models/search?${params.toString()}`,
    );
  },
  mlxUnload: () =>
    request<{ status: string; metal_cleared: boolean }>("/api/mlx/unload", {
      method: "POST",
    }),
  omlxUnloadModel: (modelId: string) =>
    request<OmlxJob & { job_id: string }>("/api/omlx/models/unload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    }),
  omlxLoadModel: (modelId: string) =>
    request<OmlxJob & { job_id: string }>("/api/omlx/models/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    }),
  omlxGetModelConfig: (repoId?: string) =>
    request<{
      repo_id: string;
      max_context_window: number | null;
      source_field: string | null;
      rope_factor: number | null;
      all_context_fields: Record<string, number>;
      config_path: string;
    }>(`/api/omlx/model-config${repoId ? `?repo_id=${encodeURIComponent(repoId)}` : ""}`),
  omlxGetCache: () => request<OmlxCacheSettings>("/api/omlx/cache"),
  omlxSetCache: (patch: Partial<OmlxCacheSettings>) =>
    request<OmlxCacheSettings>("/api/omlx/cache", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),
  omlxGetCacheStats: () => request<OmlxCacheStats>("/api/omlx/cache-stats"),
  omlxClearHotCache: () =>
    request<{ ok: boolean }>("/api/omlx/cache-stats/clear-hot", { method: "POST" }),
  omlxClearSsdCache: () =>
    request<{ ok: boolean }>("/api/omlx/cache-stats/clear-ssd", { method: "POST" }),

  // Privacy lock
  privacyStatus: () => request<PrivacyStatus>("/api/privacy/status"),
  privacyEngage: () =>
    request<PrivacyStatus>("/api/privacy/engage", { method: "POST" }),
  privacyDisengage: () =>
    request<PrivacyStatus>("/api/privacy/disengage", { method: "POST" }),
  privacyUpdate: (patch: Partial<{ allow_loopback: boolean; allow_mdns: boolean; allowed_hosts: string[]; local_only_providers: string[]; pf_anchor: string }>) =>
    request<PrivacyStatus>("/api/privacy", { method: "PUT", body: JSON.stringify(patch) }),
  privacyPfTemplate: () =>
    request<{ anchor: string; install_command: string; pf_template: string }>("/api/privacy/pf-template"),
  privacyAudit: (limit = 50) =>
    request<{ events: PrivacyAuditEntry[] }>(`/api/privacy/audit?limit=${limit}`),
  privacyCheckProvider: (provider: string) =>
    request<{ allowed: boolean; reason?: string }>("/api/privacy/check-provider", {
      method: "POST",
      body: JSON.stringify({ provider }),
    }),

  // EXO remotes
  listExoRemotes: () => request<{ remotes: ExoRemote[] }>("/api/exo/remotes"),
  addExoRemote: (data: { ssh_alias: string; label?: string; app_data_dir?: string; enabled?: boolean }) =>
    request<{ status: string; remotes: ExoRemote[] }>("/api/exo/remotes", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  updateExoRemote: (alias: string, data: Partial<{ label: string; app_data_dir: string; enabled: boolean }>) =>
    request<{ status: string; remote: ExoRemote }>(`/api/exo/remotes/${encodeURIComponent(alias)}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),
  removeExoRemote: (alias: string) =>
    request<{ status: string; remaining: number }>(`/api/exo/remotes/${encodeURIComponent(alias)}`, {
      method: "DELETE",
    }),
  exoRemoteUp: (alias: string, force = false) =>
    request<ExoJob>(`/api/exo/remotes/${encodeURIComponent(alias)}/up`, {
      method: "POST",
      body: JSON.stringify({ force }),
    }),
  exoRemoteDown: (alias: string) =>
    request<ExoJob>(`/api/exo/remotes/${encodeURIComponent(alias)}/down`, { method: "POST" }),
  exoRemoteSmoke: (alias: string) =>
    request<ExoJob>(`/api/exo/remotes/${encodeURIComponent(alias)}/smoke`, { method: "POST" }),
  exoRemoteStatus: (alias: string) =>
    request<ExoJob>(`/api/exo/remotes/${encodeURIComponent(alias)}/status`, { method: "GET" }),

  // EXO discovery
  exoDiscoverSshConfig: () =>
    request<{ hosts: SshConfigHost[] }>("/api/exo/discover/ssh-config"),
  exoDiscoverLan: (timeout = 3.0) =>
    request<{ hosts: LanSshHost[]; timeout: number }>(
      `/api/exo/discover/lan?timeout=${timeout}`,
    ),
  exoTestSsh: (alias: string, timeout = 6.0) =>
    request<{
      ok: boolean;
      return_code: number;
      stdout: string;
      stderr: string;
      hint: string;
    }>(`/api/exo/discover/test-ssh?alias=${encodeURIComponent(alias)}&timeout=${timeout}`),
  exoTbLink: () => request<ExoTbLinkSnapshot>("/api/exo/discover/tb-link"),

  // EXO setup wizard. The /setup/install-pubkey call is the ONLY path
  // that ever transmits a password — the body is constructed inline in
  // the wizard and discarded on resolve/reject. Don't add logging
  // around this fetch or store the password anywhere else.
  exoSetupLocalUser: () => request<ExoSetupLocalUser>("/api/exo/setup/local-user"),
  exoSetupProbe: (data: { host: string; user: string; port?: number; timeout?: number }) =>
    request<SshProbeResult>("/api/exo/setup/probe", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  exoSetupListKeypairs: () =>
    request<{ keypairs: LocalKeypair[] }>("/api/exo/setup/keypairs"),
  exoSetupCreateKeypair: (data: { name: string; key_type?: string; comment?: string }) =>
    request<LocalKeypair>("/api/exo/setup/keypairs", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  exoSetupInstallPubkey: (data: {
    host: string;
    user: string;
    port?: number;
    password: string;
    public_key_path: string;
    private_key_path?: string;
    timeout?: number;
  }) =>
    request<InstallPubkeyResult>("/api/exo/setup/install-pubkey", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  exoSetupAppendSshConfig: (data: {
    alias: string;
    hostname: string;
    user?: string;
    port?: number;
    identity_file?: string;
    extra_options?: Record<string, string>;
    replace?: boolean;
  }) =>
    request<SshConfigAppendResult>("/api/exo/setup/ssh-config", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  exoModels: () => request<ExoModelsResponse>("/api/exo/models"),
  exoPreloadModel: (model: string, minNodes?: number) =>
    request<ExoPreloadResult>("/api/exo/models/preload", {
      method: "POST",
      body: JSON.stringify(
        minNodes !== undefined ? { model, min_nodes: minNodes } : { model },
      ),
    }),

  // ─── Cluster-aware catalog + async preload ─────────────────────
  exoCatalog: (params?: {
    min_nodes?: number;
    ctx_len?: number;
    kv_bits?: number | null;
    refresh?: boolean;
  }) => {
    const sp = new URLSearchParams();
    if (params?.min_nodes) sp.set("min_nodes", String(params.min_nodes));
    if (params?.ctx_len) sp.set("ctx_len", String(params.ctx_len));
    if (params?.kv_bits != null) sp.set("kv_bits", String(params.kv_bits));
    if (params?.refresh) sp.set("refresh", "true");
    const q = sp.toString();
    return request<ExoCatalogResponse>(`/api/exo/catalog${q ? `?${q}` : ""}`);
  },
  exoPreloadStart: (model: string, minNodes?: number, timeout?: number) =>
    request<ExoPreloadJob>("/api/exo/preload", {
      method: "POST",
      body: JSON.stringify({
        model,
        ...(minNodes !== undefined ? { min_nodes: minNodes } : {}),
        ...(timeout !== undefined ? { timeout } : {}),
      }),
    }),
  exoPreloadStatus: (jobId: string) =>
    request<ExoPreloadJob>(`/api/exo/preload/${encodeURIComponent(jobId)}`),
  exoPreloadList: () =>
    request<{ jobs: ExoPreloadJob[] }>("/api/exo/preloads"),
  exoPreloadCancel: (jobId: string) =>
    request<{ status: string; job_id: string }>(
      `/api/exo/preload/${encodeURIComponent(jobId)}/cancel`,
      { method: "POST" },
    ),
  exoUnload: (model: string) =>
    request<{ status: string; model: string; instances_removed: number }>(
      "/api/exo/unload",
      { method: "POST", body: JSON.stringify({ model }) },
    ),

  // Lifecycle
  shutdown: () =>
    request<{ status: string }>("/api/shutdown", { method: "POST" }),

  // Unified runs
  listRuns: (params?: {
    status?: string;
    agent?: string;
    source?: string;
    search?: string;
    schedule_id?: string;
    date_from?: string;
    date_to?: string;
    order_by?: string;
    order?: "asc" | "desc";
    limit?: number;
    offset?: number;
  }) => {
    const sp = new URLSearchParams();
    if (params?.status) sp.set("status", params.status);
    if (params?.agent) sp.set("agent", params.agent);
    if (params?.source) sp.set("source", params.source);
    if (params?.search) sp.set("search", params.search);
    if (params?.schedule_id) sp.set("schedule_id", params.schedule_id);
    if (params?.date_from) sp.set("date_from", params.date_from);
    if (params?.date_to) sp.set("date_to", params.date_to);
    if (params?.order_by) sp.set("order_by", params.order_by);
    if (params?.order) sp.set("order", params.order);
    if (params?.limit != null) sp.set("limit", String(params.limit));
    if (params?.offset != null) sp.set("offset", String(params.offset));
    const qs = sp.toString();
    return request<RunsResponse>(`/api/runs${qs ? `?${qs}` : ""}`);
  },
  getRunStats: (
    period: "24h" | "7d" | "30d" | "all" | "custom" = "7d",
    dateFrom?: string,
    dateTo?: string,
    search?: string,
    status?: string,
    source?: string,
  ) => {
    const sp = new URLSearchParams({ period });
    if (dateFrom) sp.set("date_from", dateFrom);
    if (dateTo) sp.set("date_to", dateTo);
    if (search) sp.set("search", search);
    if (status) sp.set("status", status);
    if (source) sp.set("source", source);
    return request<RunStats>(`/api/runs/stats?${sp.toString()}`);
  },
  getRunEvaluation: (sessionId: string) =>
    request<RunEvaluation>(`/api/runs/${encodeURIComponent(sessionId)}/evaluation`),
  runRunEvaluation: (sessionId: string) =>
    request<RunEvaluation>(`/api/runs/${encodeURIComponent(sessionId)}/evaluation`, {
      method: "POST",
    }),
  runAgain: (sessionId: string, prompt?: string) =>
    request<{ session_id: string }>(`/api/runs/${encodeURIComponent(sessionId)}/run-again`, {
      method: "POST",
      ...(prompt !== undefined ? { body: JSON.stringify({ prompt }) } : {}),
    }),

  // Sessions
  listSessions: () => request<SessionInfo[]>("/api/sessions"),
  getSession: (id: string) =>
    request<SessionInfo>(`/api/sessions/${id}`),
  createSession: (data: Record<string, unknown>) =>
    request<SessionInfo>("/api/sessions", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  getSessionMessages: (id: string) =>
    request<Record<string, unknown>[]>(`/api/sessions/${id}/messages`),
  getSessionTimeline: (id: string) =>
    request<SessionTimeline>(`/api/sessions/${id}/timeline`),
  getSessionStatus: (id: string) =>
    request<{ active: boolean; running: boolean }>(`/api/sessions/${id}/status`),
  stopSession: (id: string) =>
    request<{ status: string }>(`/api/sessions/${id}/stop`, { method: "POST" }),
  closeSession: (id: string) =>
    request<{ status: string }>(`/api/sessions/${id}`, { method: "DELETE" }),
  clearAllSessions: () =>
    request<{ status: string; count: number }>("/api/sessions", { method: "DELETE" }),
  listSessionFiles: (id: string) =>
    request<{ path: string; size: number; modified_at: number }[]>(
      `/api/sessions/${id}/files`,
    ),
  getSessionFileUrl: (id: string, filePath: string) =>
    `${API_BASE}/api/sessions/${id}/files/${filePath.split("/").map(encodeURIComponent).join("/")}`,
  deleteSessionFile: (id: string, filePath: string) =>
    request<{ status: string }>(`/api/sessions/${id}/files/${filePath}`, {
      method: "DELETE",
    }),
  openSessionFilesFolder: (id: string, filePath?: string) =>
    request<{ status: string; path: string }>(
      `/api/sessions/${id}/files/open-folder${filePath ? `?path=${encodeURIComponent(filePath)}` : ""}`,
      { method: "POST" },
    ),
  uploadSessionFile: async (
    id: string,
    filePath: string,
    file: File,
  ): Promise<{ status: string; path: string; size: number }> => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(
      `${API_BASE}/api/sessions/${id}/files/${filePath}`,
      { method: "POST", body: form },
    );
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  },

  createSessionLink: async (
    id: string,
    source: string,
  ): Promise<{ path: string; is_dir: boolean; source: string }> => {
    const res = await fetch(`${API_BASE}/api/sessions/${id}/links`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    });
    if (!res.ok) throw new Error(`Link failed: ${res.status}`);
    return res.json();
  },

  // ── Activity timeline ────────────────────────────────────────────
  getActivityStatus: async (): Promise<{
    enabled: boolean;
    interval_secs: number;
    retain_days: number;
    exclude_apps: string[];
    running: boolean;
    db_size_bytes: number;
    max_db_mb: number;
  }> => {
    const res = await fetch(`${API_BASE}/api/activity/status`);
    if (!res.ok) throw new Error(`Activity status failed: ${res.status}`);
    return res.json();
  },

  searchActivity: async (params: {
    q?: string;
    date_from?: number;
    date_to?: number;
    app?: string;
    limit?: number;
    offset?: number;
    order_by?: "ts" | "rank";
  }): Promise<{ rows: ActivityRow[]; count: number; total: number }> => {
    const sp = new URLSearchParams();
    if (params.q) sp.set("q", params.q);
    if (params.date_from) sp.set("date_from", String(params.date_from));
    if (params.date_to) sp.set("date_to", String(params.date_to));
    if (params.app) sp.set("app", params.app);
    if (params.limit) sp.set("limit", String(params.limit));
    if (params.offset) sp.set("offset", String(params.offset));
    if (params.order_by) sp.set("order_by", params.order_by);
    const res = await fetch(`${API_BASE}/api/activity/search?${sp.toString()}`);
    if (!res.ok) throw new Error(`Activity search failed: ${res.status}`);
    return res.json();
  },

  getActivityTimeline: async (
    date?: string,
  ): Promise<{
    date: string;
    rows: ActivityRow[];
    summary: {
      from: number;
      to: number;
      total_seconds: number;
      apps: { app: string; seconds: number; samples: number }[];
    };
  }> => {
    const sp = date ? `?date=${encodeURIComponent(date)}` : "";
    const res = await fetch(`${API_BASE}/api/activity/timeline${sp}`);
    if (!res.ok) throw new Error(`Activity timeline failed: ${res.status}`);
    return res.json();
  },

  clearActivity: async (): Promise<{ deleted: number }> => {
    const res = await fetch(`${API_BASE}/api/activity/all`, { method: "DELETE" });
    if (!res.ok) throw new Error(`Activity clear failed: ${res.status}`);
    return res.json();
  },

  // ---------------------------------------------------------------------------
  // Ambient assistant
  // ---------------------------------------------------------------------------

  ambientStatus: () =>
    request<{
      enabled: boolean;
      pending_hints: number;
      allow_auto_run: boolean;
      interval_mins: number;
      llm_family: string;
      mlx_model: string;
      sweep_running: boolean;
    }>("/api/ambient/status"),

  ambientHints: () =>
    request<{ hints: import("../types").AmbientHint[]; quiet_hours: boolean }>(
      "/api/ambient/hints",
    ),

  ambientRun: () =>
    request<{ status: string; hints_added: number; skipped: string | null }>(
      "/api/ambient/run",
      { method: "POST" },
    ),

  ambientAccept: (
    hintId: string,
    mode: "chat" | "run" | "apply" = "chat",
    agentName?: string,
  ) =>
    request<{
      status: string;
      session_id?: string | null;
      target_kind?: string;
      target_id?: string;
    }>(
      `/api/ambient/hints/${hintId}/accept`,
      {
        method: "POST",
        body: JSON.stringify({ mode, agent_name: agentName ?? null }),
      },
    ),

  ambientDismiss: (hintId: string) =>
    request<{ status: string }>(
      `/api/ambient/hints/${hintId}/dismiss`,
      { method: "POST" },
    ),

  ambientSnooze: (hintId: string, hours = 4) =>
    request<{ status: string; hours: number }>(
      `/api/ambient/hints/${hintId}/snooze`,
      { method: "POST", body: JSON.stringify({ hours }) },
    ),

  // Voice
  voiceCatalog: () =>
    request<{ rows: import("../types").VoiceCatalogRow[]; enriching: boolean }>("/api/voice/catalog"),
  voiceStatus: () =>
    request<import("../types").VoiceStatus>("/api/voice/status"),
  voiceSttTest: (audio_b64: string, language?: string) =>
    request<{ text?: string; error?: string }>(
      "/api/voice/stt-test",
      { method: "POST", body: JSON.stringify({ audio_b64, language }) },
    ),
  voiceModelCached: (repoId: string) =>
    request<{ repo_id: string; cached: boolean }>(`/api/voice/model-cached?repo_id=${encodeURIComponent(repoId)}`),
  loopbackStatus: () =>
    request<import("../types").LoopbackStatus>("/api/voice/loopback-status"),

  // ── Screen capture (Transcribe screenshots) ───────────────────────
  captureScreenPermission: () =>
    request<import("../types").CapturePermission>("/api/capture/permission"),
  captureScreenPromptPermission: () =>
    request<import("../types").CapturePermission>("/api/capture/permission/prompt", {
      method: "POST",
    }),
  captureWindows: (thumbnails = false) =>
    request<{ supported: boolean; windows: import("../types").CaptureWindow[] }>(
      `/api/capture/windows${thumbnails ? "?thumbnails=true" : ""}`,
    ),
  captureScreen: (
    mode: "desktop" | "window",
    windowId?: number,
    lastHash?: string,
  ) =>
    request<import("../types").CaptureResult>("/api/capture/screen", {
      method: "POST",
      body: JSON.stringify({ mode, window_id: windowId, last_hash: lastHash }),
    }),
};

export interface ActivityRow {
  id: number;
  ts: number;
  app: string;
  title: string;
  url: string;
  file_path: string;
  context: string;
  duration_s: number;
}
