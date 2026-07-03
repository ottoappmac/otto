"""Pydantic models for the REST / WebSocket API."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# WebSocket message types
# ---------------------------------------------------------------------------

class WSMessageType(str, Enum):
    USER = "user"
    AGENT = "agent"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATUS = "status"
    ERROR = "error"
    DONE = "done"


class WSMessage(BaseModel):
    type: WSMessageType
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Agent specs
# ---------------------------------------------------------------------------

class AgentSpec(BaseModel):
    name: str
    description: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    model_override: Optional[str] = None
    """When ``subagent_llm_family`` is ``custom``, passed to ``init_chat_model`` / ``resolve_model``."""

    subagent_llm_family: Optional[str] = None
    """``inherit`` | ``frontier`` | ``mlx`` | ``custom``. ``None`` = legacy (treat as custom if ``model_override`` else inherit)."""

    mlx_model_id: Optional[str] = None
    """Optional HF repo id when ``subagent_llm_family`` is ``mlx``; defaults to global MLX text model."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    builtin: bool = False
    """True when this agent is shipped with the app (managed by ``seed_defaults``).
    Built-ins cannot be deleted via the REST API; the field is computed on read
    and never persisted to disk so it stays in sync with the registry."""


class AgentCreateRequest(BaseModel):
    name: str
    description: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    model_override: Optional[str] = None
    subagent_llm_family: Optional[str] = None
    mlx_model_id: Optional[str] = None


class AgentGenerateRequest(BaseModel):
    """Ask the LLM to generate an agent spec from a natural-language description."""
    user_description: str


# ---------------------------------------------------------------------------
# Skill specs
# ---------------------------------------------------------------------------

class SkillSpec(BaseModel):
    name: str
    description: str
    content: str  # full SKILL.md content (frontmatter + body)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    builtin: bool = False
    """True when this skill is shipped with the app (managed by ``seed_defaults``).
    Built-ins cannot be deleted via the REST API; the field is computed on read
    and never persisted to disk so it stays in sync with the registry."""


class SkillCreateRequest(BaseModel):
    name: str
    description: str
    content: str


class SkillGenerateRequest(BaseModel):
    user_description: str


# ---------------------------------------------------------------------------
# MCP server management
# ---------------------------------------------------------------------------

class MCPAuthStatus(BaseModel):
    """Per-server auth-flow snapshot the frontend renders next to Start.

    All fields are name-only / boolean — no token contents.  Generated
    on every list / start response so the UI can pick the right
    affordance (text-input dialog, "Login" button, or "Re-login"
    button) without re-fetching anything.
    """

    kind: str = "static"
    has_bundle: bool = False
    expired: bool = False
    needs_login: bool = False
    expiry_iso: Optional[str] = None


class MCPServerStatus(BaseModel):
    id: str
    name: str
    connected: bool = False
    tool_count: int = 0
    tools: list[str] = Field(default_factory=list)
    excluded_tools: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    auto_start: bool = False
    process_running: bool = False
    transport: str = "streamable_http"
    url: Optional[str] = None
    port: Optional[int] = None
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    builtin: bool = False
    requires_os: Optional[str] = None
    os_supported: bool = True
    server_type: str = "generic"
    context_cache_active: bool = False
    # Agent-built MCP plumbing.  ``generated`` flags servers authored by
    # the agent's mcp_builder pipeline; the UI uses it to (a) badge them
    # as "Agent-built" and (b) route delete through the registry which
    # also wipes the source file and vault entries.  ``required_secrets``
    # is the full list of credential names the subprocess needs in env;
    # ``missing_secrets`` is the subset that aren't yet stored in the
    # OS keychain — the UI uses this to disable "Start" with a "Set
    # credentials first" affordance instead of letting the subprocess
    # fail at spawn.
    generated: bool = False
    required_secrets: list[str] = Field(default_factory=list)
    missing_secrets: list[str] = Field(default_factory=list)
    # Optional credentials surface in the credentials dialog so users
    # can set / update them, but never block the Start button.  See
    # ``MCPServerConfig.optional_secrets`` for the storage definition.
    optional_secrets: list[str] = Field(default_factory=list)
    # Interactive auth status (OAuth / browser-capture).  ``kind ==
    # "static"`` means no flow — the credentials dialog renders the
    # text-input list as before.  Otherwise ``needs_login`` drives a
    # "Login" button that POSTs ``/api/mcp-servers/{id}/auth/login``.
    auth: MCPAuthStatus = Field(default_factory=MCPAuthStatus)


class MCPServerAddRequest(BaseModel):
    name: str
    transport: str = "streamable_http"
    url: Optional[str] = None
    port: Optional[int] = None
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    auto_start: bool = False


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class SessionInfo(BaseModel):
    id: str
    agent_name: Optional[str] = None
    title: str = "New Session"
    message_count: int = 0
    tools_used: list[str] = Field(default_factory=list)
    schedule_id: Optional[str] = None
    trigger_source: Optional[str] = None
    # Child-session linkage.  When the orchestrator calls
    # ``spawn_followup_session`` to hand off a request to a fresh graph
    # (typically after creating new MCP tools or sub-agents), the new
    # session records its parent here.  ``chain_depth`` tracks how deep
    # the spawn chain runs and is used by ``SessionManager.spawn_child_session``
    # to enforce a hard cap and prevent runaway loops.
    parent_session_id: Optional[str] = None
    chain_depth: int = 0
    # Set when the session was spawned by a custom trigger
    # (``trigger_source == "trigger"``) or the ambient assistant
    # (``trigger_source == "ambient"``).  History UI uses this to
    # link back to the originating trigger or hint.
    trigger_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Run metrics — populated during/after streaming
    status: str = "idle"  # idle | running | completed | error | stopped | awaiting_input
    finished_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    llm_provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    error: Optional[str] = None
    # Coarse error classification (e.g. ``llm_rate_limit``, ``internal``) used by
    # the errored-run analyzer to decide whether a prompt fix could help.
    error_code: Optional[str] = None
    # MLX throughput stats (token-weighted averages over the session).
    # Null for non-MLX providers that don't report generation telemetry.
    avg_prefill_tps: Optional[float] = None  # "TIPS" — prefill tokens/sec
    avg_generation_tps: Optional[float] = None  # "TOPS" — generation tokens/sec
    cache_hit_ratio: Optional[float] = None  # KV cache reuse ratio (0..1)
    peak_memory_gb: Optional[float] = None  # peak GPU memory during session
    # End-of-run evaluation summary (full results live in {id}.eval.json).
    # ``eval_status``: none | running | done | skipped | error.
    eval_status: Optional[str] = None
    eval_overall_score: Optional[float] = None  # mean metric score (0..1)
    eval_pass_count: Optional[int] = None  # metrics that met their threshold
    eval_total: Optional[int] = None  # total scored metrics


class SessionCreateRequest(BaseModel):
    agent_name: Optional[str] = None
    trigger_source: Optional[str] = None


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

_DEFAULT_SCHEDULE_TIMEOUT = 24 * 60 * 60  # 24 hours


class ScheduleSpec(BaseModel):
    id: str
    agent_name: Optional[str] = None
    prompt: str
    cron_expression: str
    enabled: bool = True
    keep_last_n_runs: int = 30
    timeout_seconds: int = _DEFAULT_SCHEDULE_TIMEOUT
    last_run: Optional[datetime] = None
    last_status: Optional[str] = None
    last_error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScheduleCreateRequest(BaseModel):
    id: str
    agent_name: Optional[str] = None
    prompt: str
    cron_expression: str


class ScheduleUpdateRequest(BaseModel):
    prompt: Optional[str] = None
    cron_expression: Optional[str] = None
    enabled: Optional[bool] = None
    keep_last_n_runs: Optional[int] = None
    agent_name: Optional[str] = None


class ScheduleRun(BaseModel):
    id: str
    schedule_id: str
    status: str  # "running" | "success" | "error" | "cancelled"
    started_at: datetime
    finished_at: Optional[datetime] = None
    message_count: int = 0
    error: Optional[str] = None
    session_id: Optional[str] = None
    # End-of-run evaluation summary, surfaced from the linked session's meta.
    # none | running | done | skipped | error.
    eval_status: Optional[str] = None
    eval_overall_score: Optional[float] = None  # mean metric score (0..1)
    eval_pass_count: Optional[int] = None
    eval_total: Optional[int] = None


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------
#
# A trigger is a user-defined polling rule that fires an agent when a
# condition becomes true.  Two condition types are supported today:
#
# * ``fileos``    — watch a path on the filesystem.  Sub-modes:
#                   ``mtime`` / ``size`` / ``exists`` / ``new_files``.
# * ``macostool`` — run an osascript snippet and compare its output.
#
# When the condition fires, the trigger manager spawns a session for
# the configured agent with ``trigger_source="trigger"`` and a prompt
# that includes the event payload so the worker agent has full context.

_DEFAULT_TRIGGER_TIMEOUT = 24 * 60 * 60  # 24 hours
_DEFAULT_POLL_SECONDS = 60
_MIN_POLL_SECONDS = 5
_MAX_POLL_SECONDS = 24 * 60 * 60

TriggerType = Literal["fileos", "macostool", "http", "git", "shell"]
FileOsWatch = Literal["mtime", "size", "exists", "new_files"]
OsaLanguage = Literal["AppleScript", "JavaScript"]
HttpMode = Literal["status_change", "body_hash", "json_value", "regex"]
HttpMethod = Literal["GET", "POST", "HEAD"]
ShellMode = Literal["stdout_change", "regex", "exit_code_change"]


class TriggerSpec(BaseModel):
    """Persisted trigger definition.

    ``state_json`` carries the per-trigger watermark the manager needs
    to detect change between polls (last mtime, last seen file list,
    sha256 of last stdout, …).  It's read-write at runtime and must
    survive backend restarts, hence persisted on the spec rather than
    in a separate sidecar file.
    """

    id: str
    type: TriggerType
    poll_seconds: int = _DEFAULT_POLL_SECONDS
    agent_name: Optional[str] = None
    prompt: str
    enabled: bool = True

    # fileos
    path: Optional[str] = None
    watch: FileOsWatch = "mtime"
    glob: Optional[str] = None

    # macostool
    script: Optional[str] = None
    language: OsaLanguage = "AppleScript"
    match: Optional[str] = None  # optional regex applied to stdout (also: http/shell)

    # http
    url: Optional[str] = None
    http_mode: HttpMode = "body_hash"
    method: HttpMethod = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    body: Optional[str] = None
    json_path: Optional[str] = None  # dotted path, e.g. "data.items.0.id"

    # git
    repo_path: Optional[str] = None
    branch: str = "HEAD"
    author_filter: Optional[str] = None  # regex on author email/name
    path_filter: Optional[str] = None    # only fire if commits touched this glob

    # shell
    command: Optional[str] = None
    shell_mode: ShellMode = "stdout_change"
    cwd: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)

    state_json: dict[str, Any] = Field(default_factory=dict)

    # True for triggers seeded from the managed catalog — cannot be deleted
    # via the API; can only be enabled/disabled and have their prompt/agent
    # updated.
    builtin: bool = False

    keep_last_n_runs: int = 30
    timeout_seconds: int = _DEFAULT_TRIGGER_TIMEOUT
    last_run: Optional[datetime] = None
    last_status: Optional[str] = None
    last_error: Optional[str] = None
    last_event: Optional[dict[str, Any]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TriggerCreateRequest(BaseModel):
    id: str
    type: TriggerType
    prompt: str
    poll_seconds: int = _DEFAULT_POLL_SECONDS
    agent_name: Optional[str] = None
    # fileos
    path: Optional[str] = None
    watch: FileOsWatch = "mtime"
    glob: Optional[str] = None
    # macostool
    script: Optional[str] = None
    language: OsaLanguage = "AppleScript"
    match: Optional[str] = None
    # http
    url: Optional[str] = None
    http_mode: HttpMode = "body_hash"
    method: HttpMethod = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    body: Optional[str] = None
    json_path: Optional[str] = None
    # git
    repo_path: Optional[str] = None
    branch: str = "HEAD"
    author_filter: Optional[str] = None
    path_filter: Optional[str] = None
    # shell
    command: Optional[str] = None
    shell_mode: ShellMode = "stdout_change"
    cwd: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)


class TriggerUpdateRequest(BaseModel):
    prompt: Optional[str] = None
    poll_seconds: Optional[int] = None
    enabled: Optional[bool] = None
    agent_name: Optional[str] = None
    # fileos
    path: Optional[str] = None
    watch: Optional[FileOsWatch] = None
    glob: Optional[str] = None
    # macostool
    script: Optional[str] = None
    language: Optional[OsaLanguage] = None
    match: Optional[str] = None
    # http
    url: Optional[str] = None
    http_mode: Optional[HttpMode] = None
    method: Optional[HttpMethod] = None
    headers: Optional[dict[str, str]] = None
    body: Optional[str] = None
    json_path: Optional[str] = None
    # git
    repo_path: Optional[str] = None
    branch: Optional[str] = None
    author_filter: Optional[str] = None
    path_filter: Optional[str] = None
    # shell
    command: Optional[str] = None
    shell_mode: Optional[ShellMode] = None
    cwd: Optional[str] = None
    env: Optional[dict[str, str]] = None
    keep_last_n_runs: Optional[int] = None


class TriggerRun(BaseModel):
    id: str
    trigger_id: str
    status: str  # "running" | "success" | "error" | "cancelled"
    started_at: datetime
    finished_at: Optional[datetime] = None
    message_count: int = 0
    error: Optional[str] = None
    session_id: Optional[str] = None
    event_payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestConnectionRequest(BaseModel):
    provider: str
    api_key: str = ""
    model_name: str = ""
    model_provider: str = "anthropic"
    bedrock_region: str = "us-east-1"
    bedrock_auth_mode: str = "keys"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    # MLX / Hugging Face Hub (when provider == "mlx")
    hf_llm_model_id: str = ""
    hf_vlm_model_id: str = ""
    hf_draft_llm_model_id: str = ""
    hf_token: str = ""
    # OpenAI — native or Azure (when provider == "openai")
    openai_model_provider: str = "openai"   # "openai" | "azure"
    azure_endpoint: str = ""
    azure_api_version: str = "2024-12-01-preview"
    azure_deployment: str = ""


class TestConnectionResponse(BaseModel):
    success: bool
    message: str = ""
    models: list[dict] = []
