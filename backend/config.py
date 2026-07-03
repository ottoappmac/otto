"""Application configuration management.

Replaces .env-based config with structured JSON storage.
Config lives in the platform-specific app data directory:
  macOS:   ~/Library/Application Support/Otto/config.json
  Windows: %APPDATA%/Otto/config.json
  Linux:   ~/.config/Otto/config.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import uuid as _uuid

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.credential_vault import CredentialVaultError, app_vault
from backend.mlx_hub_paths import resolve_hf_hub_cache_dir
from backend.utils import platform_label

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Secret fields — stored in the OS keychain, never in plaintext config.json
#
# Each tuple is ``(dotted attribute path on AppConfig, keychain account
# name)``.  These values are scrubbed from the file on
# :meth:`AppConfig.save` (routed to ``app_vault``) and re-hydrated onto
# the in-memory model on :meth:`AppConfig.load`.  The set is static, so
# no sidecar name index is needed (unlike the per-server MCP vault).
# ---------------------------------------------------------------------------

_SECRET_FIELDS: tuple[tuple[str, str], ...] = (
    ("llm.anthropic.api_key", "anthropic_api_key"),
    ("llm.anthropic.aws_access_key_id", "aws_access_key_id"),
    ("llm.anthropic.aws_secret_access_key", "aws_secret_access_key"),
    ("llm.openai.api_key", "openai_api_key"),
    ("llm.openai.azure_api_key", "azure_api_key"),
    ("llm.mlx.hf_token", "hf_token"),
    ("observability.langsmith.api_key", "langsmith_api_key"),
    ("omlx.admin_api_key", "omlx_admin_api_key"),
)

_vault_unavailable_warned = False


def _warn_vault_unavailable_once() -> None:
    global _vault_unavailable_warned
    if not _vault_unavailable_warned:
        logger.warning(
            "Credential vault unavailable — storing secrets in plaintext "
            "config.json. Install/enable a keyring backend to secure them."
        )
        _vault_unavailable_warned = True


def _model_get_path(obj: object, path: str) -> object:
    cur: object = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _model_set_path(obj: object, path: str, value: object) -> None:
    parts = path.split(".")
    cur: object = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def _dict_get_path(data: dict, path: str) -> object:
    cur: object = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _dict_set_path(data: dict, path: str, value: object) -> None:
    parts = path.split(".")
    cur: object = data
    for part in parts[:-1]:
        nxt = cur.get(part) if isinstance(cur, dict) else None
        if not isinstance(nxt, dict):
            return
        cur = nxt
    if isinstance(cur, dict) and parts[-1] in cur:
        cur[parts[-1]] = value


def get_app_data_dir() -> Path:
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    # Migrate existing config from the old "George" directory if present
    # and the new "Otto" directory does not yet exist.
    old_path = base / "George"
    new_path = base / "Otto"
    if old_path.exists() and not new_path.exists():
        import shutil
        shutil.move(str(old_path), str(new_path))
    return new_path


# ---------------------------------------------------------------------------
# Token pricing table  (USD per 1 000 000 tokens, input/output)
# ---------------------------------------------------------------------------

_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-3": (0.25, 1.25),
    "claude-haiku-3-5": (0.8, 4.0),
    # OpenAI
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    "o1": (15.0, 60.0),
    "o1-mini": (1.1, 4.4),
    "o3": (10.0, 40.0),
    "o3-mini": (1.1, 4.4),
    "o4-mini": (1.1, 4.4),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return an estimated USD cost from a pricing table.

    Tries exact match first, then prefix match (e.g. ``claude-sonnet-4-6``
    matched by ``claude-sonnet-4``).  Returns 0.0 for unknown / local models.
    """
    key = model.lower()
    if key in _PRICING:
        inp, out = _PRICING[key]
    else:
        inp, out = 0.0, 0.0
        for prefix, rates in _PRICING.items():
            if key.startswith(prefix):
                inp, out = rates
                break
    if inp == 0.0 and out == 0.0:
        return 0.0
    return (input_tokens * inp + output_tokens * out) / 1_000_000


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

class AnthropicConfig(BaseModel):
    model_provider: str = "anthropic"
    api_key: str = ""
    model_name: str = "claude-sonnet-4-6"
    bedrock_region: str = "us-east-1"
    bedrock_auth_mode: str = "keys"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    max_tokens: int = 16384
    thinking_enabled: bool = False
    thinking_budget: int = 2048
    tool_efficient: bool = True


class OpenAIConfig(BaseModel):
    model_provider: str = "openai"  # "openai" | "azure"
    api_key: str = ""               # native OpenAI API key (sk-...)
    azure_api_key: str = ""         # Azure OpenAI API key (separate from native)
    model_name: str = "gpt-4o"
    azure_endpoint: str = ""
    azure_api_version: str = "2024-12-01-preview"
    azure_deployment: str = ""
    max_tokens: int = 16384
    temperature: float = 0.0


class MlxBookmark(BaseModel):
    """Optional display label for a cached Hub repo (shown in the MLX model pickers)."""

    repo_id: str
    label: str = ""


class MlxHfConfig(BaseModel):
    """Hugging Face Hub identifiers for local MLX inference (``LLM_PROVIDER=mlx``)."""

    hf_llm_model_id: str = "mlx-community/quantized-gemma-2b-it"
    hf_vlm_model_id: str = ""
    hf_draft_llm_model_id: str = ""
    hf_token: str = ""
    # When empty, Hugging Face uses its default (~/.cache/huggingface/hub). Same as env HF_HUB_CACHE.
    # Relative to ~/.cache/ unless absolute or ~-prefixed (e.g. huggingface/hub).
    hf_hub_cache: str = "huggingface/hub"
    mlx_bookmarks: list[MlxBookmark] = Field(default_factory=list)
    # Generation / KV (mirrors ``utilities.environment.Environment`` MLX_* keys)
    mlx_max_tokens: int = 512
    mlx_temp: float = 0.0
    mlx_verbose: bool = False
    mlx_thinking: bool = False
    mlx_prompt_cache: bool = False
    mlx_system_prompt_cache: bool = False
    mlx_kv_bits: Optional[int] = None  # None = full precision; 4 or 8 = quantized KV
    mlx_kv_group_size: int = 64  # quantization group size — smaller = better quality, more overhead
    mlx_repetition_penalty: float = 1.1
    mlx_num_draft_tokens: int = 3  # tokens proposed per speculative step; tune to model pair
    # Soft cap on the KV prefix cache, in tokens.  After each generation the
    # chat model checks the cumulative cache offset; if it exceeds this value,
    # the cache is trimmed down to ~half so the next turn doesn't immediately
    # trip the cap again.  This is the primary defence against unbounded
    # memory growth in long autonomous sessions — a 7B 4-bit model uses
    # roughly 32 KB of cache per token, so the default 32 768 tokens caps the
    # KV pool at ~1 GB (smaller for sub-7B / lower-precision models).  Set
    # to 0 to disable the cap (legacy unbounded behaviour).
    mlx_prompt_cache_max_tokens: int = 32768

    # ── Turbo mode (oMLX-derived optimisations) ──────────────────────────
    #
    # Opt-in pipeline stacked on top of the classic ChatMLXText path.  The
    # level string selects progressively more aggressive optimisations; the
    # factory in :mod:`deep_agent.model_factory` falls back to the classic
    # path whenever initialisation fails, so flipping these is safe even on
    # machines that don't yet have the vendored turbo package available.
    #
    #   off    — original behaviour, no turbo engine.
    #   basic  — single-threaded MLX executor + continuous batcher.
    #   cache  — basic + paged / prefix KV cache.
    #   ssd    — cache + SSD cold tier (cross-session KV persistence).
    #   max    — ssd + TurboQuant 4-bit KV with Metal attention kernel.
    turbo_level: str = "off"
    # Directory for the SSD cold-tier KV cache.  Empty ⇒ auto-pick
    # ``<app_data>/kv_cache`` at turbo init time.
    turbo_ssd_dir: str = ""
    # Soft cap on the SSD cache footprint (GB).  LRU-evicted beyond this.
    turbo_ssd_max_gb: int = 50
    # TurboQuant KV bits (4 or 8).  Only used when ``turbo_level == "max"``.
    turbo_tq_bits: int = 4
    # Paged-cache block size (tokens per block).  Default aligned with the
    # omlx engine's recommended value.
    turbo_block_size: int = 256


class HookDefinition(BaseModel):
    """A single hook definition for Claude Code or OpenClaw.

    Hooks can inject prompts, gate tool calls, or add context.  The
    ``type`` field determines which content field is used:

    - ``"prompt"`` → uses ``prompt`` (LLM-evaluated policy check)
    - ``"command"`` → uses ``command`` (shell script / binary)
    - ``"http"`` → uses ``url`` (HTTP endpoint)
    """

    id: str = Field(default_factory=lambda: _uuid.uuid4().hex[:8])
    event: str = "PreToolUse"
    type: str = "prompt"
    url: str = ""
    command: str = ""
    prompt: str = ""
    matcher: str = ""
    timeout: int = 10
    enabled: bool = True


class ClaudeHookConfig(BaseModel):
    """Configuration for Claude session evaluation.

    ``enabled`` controls the JSONL transcript reader (the eval-hook MCP).
    ``http_hooks_enabled`` activates the HTTP endpoints that Claude Code
    can POST events to for real-time push-based monitoring.
    ``quality_gate_*`` fields control the optional Stop-event gate that
    can block Claude from finishing when the tool error rate is too high.
    ``hooks`` stores user-defined hooks (prompt injection, tool gating,
    etc.) that are merged into the generated Claude Code settings snippet.
    """

    enabled: bool = True
    http_hooks_enabled: bool = False
    quality_gate_enabled: bool = False
    quality_gate_threshold: float = 0.5
    auto_monitor_enabled: bool = False
    max_auto_sessions: int = 3
    auto_monitor_agent: str = "claude-session-eval-agent"
    hooks: list[HookDefinition] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    """End-of-run evaluation of OTTO's own sessions.

    When ``auto_evaluate`` is on, each completed run is scored once at the
    end: an LLM auto-selects appropriate DeepEval metrics for the task and
    runs them in-process via ``tools.evaluation.evaluators``.  A manual
    "Evaluate" button in the UI runs the same flow on demand, but is only
    surfaced as enabled when ``auto_evaluate`` is off.

    ``llm_family`` controls which model judges (mirrors ``MemoryConfig``):
    ``follow_main`` reuses the main chat provider, ``frontier`` forces
    Anthropic / Bedrock, ``mlx`` forces a local MLX model.
    """

    auto_evaluate: bool = False
    # When on, a run that ends in ``error`` is analyzed: the failure is
    # classified and, only for prompt-addressable errors, the judge LLM drafts
    # a stronger user prompt (surfaced on the Evaluation tab + Suggestions inbox).
    analyze_errors: bool = False
    llm_family: str = "follow_main"  # follow_main | frontier | mlx
    max_metrics: int = 4
    threshold: float = 0.5


class OpenClawConfig(BaseModel):
    """Configuration for OpenClaw transcript access.

    Supports two modes:
    - **local**: reads ``~/.openclaw/`` directly from the local filesystem.
    - **ssh**: reads files from a remote host via SSH (e.g. AWS Lightsail).

    The **session watcher** periodically scans the sessions directory for
    new ``.jsonl`` files and can auto-start eval sessions for them.
    """

    enabled: bool = False
    mode: str = "local"  # "local" | "ssh"
    state_dir: str = "~/.openclaw"
    ssh_host: str = ""
    ssh_user: str = "ubuntu"
    ssh_key_path: str = ""
    ssh_port: int = 22

    watcher_enabled: bool = False
    watcher_poll_interval: int = 10  # seconds between scans
    auto_monitor_enabled: bool = False
    max_auto_sessions: int = 3
    auto_monitor_agent: str = "openclaw-session-eval-agent"


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    mlx: MlxHfConfig = Field(default_factory=MlxHfConfig)


class OrchestratorConfig(BaseModel):
    """Separate LLM family for the Deep Agent orchestrator vs chat/subagents.

    ``llm_family`` controls ``DEEP_AGENT_LLM_PROVIDER`` / MLX overrides.
    Legacy ``provider_override`` (e.g. ``anthropic`` / ``mlx``) still wins when set.
    """

    llm_family: str = "follow_main"  # follow_main | frontier | mlx | exo
    mlx_model: str = ""  # optional HF repo id for orchestrator when llm_family is mlx
    mlx_model_type: str = "llm"  # llm | vlm — maps to DEEP_AGENT_MLX_MODEL_TYPE
    provider_override: Optional[str] = None

    # Prompt size mode for the orchestrator system prompt.  See
    # ``Environment.use_lite_orchestrator_prompt()`` for resolution rules.
    #   "auto" (default) — lite when orchestrator runs on mlx/exo, full otherwise.
    #   "full"           — force long Claude-tuned prompt regardless of provider.
    #   "lite"           — force short prompt regardless of provider.
    prompt_mode: str = "auto"  # auto | full | lite

    # LangGraph recursion limit — maximum number of graph steps per run.
    # The deepagents library default is 1 000; LangGraph's own ceiling is
    # 10 000 (or LANGGRAPH_DEFAULT_RECURSION_LIMIT env var).  Raise for
    # very long autonomous tasks; lower to catch runaway loops early.
    recursion_limit: int = 10000

    # Per-run tool-call budget (finer-grained than recursion_limit).  At the
    # soft budget the agent is nudged once to converge; at the hard budget the
    # run ends gracefully with a partial result.  0 disables either threshold.
    tool_call_soft_budget: int = 80
    tool_call_hard_budget: int = 150


class MCPAuthConfig(BaseModel):
    """How the MCP manager should obtain credentials for a server.

    Defaults to ``kind="static"`` so any existing config keeps the
    paste-a-string flow that ``required_secrets`` already covers — no
    migration needed for upgraded users.

    Other ``kind`` values dispatch to a provider in :mod:`backend.auth`
    that runs an interactive flow (OAuth device, OAuth auth-code,
    browser bearer-token capture) and persists a structured token
    bundle in the OS keychain via
    :meth:`backend.credential_vault.CredentialVault.set_bundle`.

    The MCP subprocess never sees this struct directly; the manager
    projects ``env_mapping`` over the bundle and forwards only the
    resulting flat env dict.
    """

    kind: str = "static"  # static | oauth_device | oauth_authcode | browser_capture

    # OAuth device + auth-code shared fields
    client_id: str = ""
    # Public native clients SHOULD NOT need a secret; left empty unless
    # the issuer specifically requires it.
    client_secret: str = ""
    auth_url: str = ""    # auth-code only (user consent URL)
    token_url: str = ""   # auth-code + device (token exchange / refresh)
    device_url: str = ""  # device only (start device authorization)
    scopes: list[str] = Field(default_factory=list)

    # Browser-capture (CDP) fields
    landing_url: str = ""           # where to send the user
    header_name: str = "Authorization"
    token_prefix: str = "Bearer "   # stripped before storage; "" keeps verbatim
    # Defence against a malicious manifest sending the user through a
    # phishing intermediary.  Empty allows any host (used only by
    # built-in providers; agent-authored manifests must declare an
    # explicit list — enforced at registration time).
    allowed_hosts: list[str] = Field(default_factory=list)

    # Projection: env-var name → bundle field (``access_token``,
    # ``refresh_token``, ``token_type``, ``expiry_iso``,
    # ``obtained_iso``, or ``"extra.<name>"``).  Empty dict + non-static
    # kind is a misconfiguration — the subprocess won't see anything.
    env_mapping: dict[str, str] = Field(default_factory=dict)

    # Extra query parameters appended to the authorization URL for
    # ``oauth_authcode`` flows.  Typical use: Google requires
    # ``{"access_type": "offline", "prompt": "consent"}`` to obtain a
    # refresh token from a Desktop OAuth app.  These are plain strings
    # and must never contain credential-shaped values.
    extra_auth_params: dict[str, str] = Field(default_factory=dict)

    # Self-serve dev console URL where a user (or the agent via
    # ``web-voyager`` auto-signup) can register and obtain an API key.
    # Surfaced in the credentials dialog and the ``request_credential``
    # tool so the user has one click to the right page when manual
    # paste is needed.  Optional — leave empty for MCPs whose
    # credentials are issued out-of-band (e.g. internal corporate
    # systems).
    signup_url: str = ""


class MCPServerConfig(BaseModel):
    id: str
    name: str
    transport: str = "streamable_http"
    url: Optional[str] = None
    port: Optional[int] = None
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    auto_start: bool = False
    excluded_tools: list[str] = Field(default_factory=list)
    builtin: bool = False
    requires_os: Optional[str] = None
    # Names of credentials that the manager will pull from
    # ``CredentialVault`` and inject as env vars when spawning this MCP's
    # subprocess.  The vault stores them under
    # ``service="otto.mcp.{id}", account="{name}"``.  The values never
    # touch the LLM context — they live in keychain and only enter the
    # subprocess environment at spawn time.
    required_secrets: list[str] = Field(default_factory=list)
    # Optional credentials: surfaced in the credentials dialog so the
    # user can set / update them, and hydrated into the subprocess env
    # if present, but the MCP is allowed to start without them (because
    # it has a sensible default or the credential is purely an
    # enhancement).  Built-in MCPs use this for things like
    # ``EDGAR_USER_AGENT`` where SEC requires *some* contact string but
    # we ship a working fallback.
    optional_secrets: list[str] = Field(default_factory=list)
    # When True, this server's source file lives under the agent-built
    # MCPs directory and was created by the ``mcp_builder`` tool.  Used
    # by the registry to scope deletion (config + file + creds).
    generated: bool = False
    # Interactive auth configuration.  Defaults to "static" — the
    # historical paste-a-string-into-required_secrets flow.  Anything
    # else routes through :mod:`backend.auth` and stores tokens in a
    # structured bundle (see :class:`MCPAuthConfig`).
    auth: MCPAuthConfig = Field(default_factory=MCPAuthConfig)


class LangSmithConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    endpoint: str = "https://api.smith.langchain.com"
    project: str = "Research"


class ObservabilityConfig(BaseModel):
    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)
    log_level: str = "INFO"


def _default_mcp_servers() -> list[MCPServerConfig]:
    servers = [
        MCPServerConfig(
            id="playwright-mcp",
            name="Playwright MCP (Local)",
            transport="streamable_http",
            url="http://localhost:8931/mcp",
            enabled=True,
            auto_start=True,
            builtin=True,
        ),
        MCPServerConfig(
            id="agent-eval-service",
            name="Agent Evaluator Service",
            transport="streamable_http",
            url="http://localhost:8941/mcp",
            enabled=True,
            auto_start=True,
            builtin=True,
        ),
        MCPServerConfig(
            id="claude-eval-hook",
            name="Claude Evaluator Hook",
            transport="streamable_http",
            url="http://localhost:8942/mcp",
            enabled=True,
            auto_start=True,
            builtin=True,
        ),
        MCPServerConfig(
            id="openclaw-eval-hook",
            name="OpenClaw Evaluator Hook",
            transport="streamable_http",
            url="http://localhost:8943/mcp",
            enabled=False,
            auto_start=True,
            builtin=True,
        ),
    ]

    # Source-bundled built-in MCPs (server.py + requirements.txt live in
    # the repo under ``backend/builtin_mcps/<id>/``).  The ``builtin_mcps``
    # registry knows how to mirror those files into the user's app data
    # dir and provision per-MCP venvs.  Importing here keeps config-load
    # cheap; the registry module itself imports nothing heavy.
    from backend.builtin_mcps import BUILTIN_MCPS, builtin_mcp_config

    for mcp in BUILTIN_MCPS:
        servers.append(builtin_mcp_config(mcp))

    if platform_label() == "macos":
        servers.append(
            MCPServerConfig(
                id="macos-native",
                name="macOS Desktop (Accessibility)",
                transport="stdio",
                builtin=True,
                enabled=True,
                auto_start=False,
                requires_os="macos",
            ),
        )
    return servers


class ExoRemoteConfig(BaseModel):
    """One secondary node in a multi-machine ``exo`` cluster.

    ``ssh_alias`` is a host entry from ``~/.ssh/config`` (must be reachable
    with ``ssh <alias>`` non-interactively, i.e. key-based auth, no
    passphrase prompt).  ``label`` is a friendly name shown in the UI.
    ``app_data_dir`` overrides ``$OTTO_APP_DATA_DIR`` on the remote so
    the script can land its repo / pidfiles / logs in a sensible
    location even when the remote user doesn't have a ``Otto.app``.

    ``identity_file`` is informational only — when the remote was
    bootstrapped via the Cluster setup wizard, this is the local
    private key whose public counterpart is now in the remote's
    ``authorized_keys``. The actual SSH dial still goes via ``ssh_alias``
    so that ``~/.ssh/config`` is the single source of truth.
    """

    ssh_alias: str
    label: str = ""
    app_data_dir: str = ""
    enabled: bool = True
    identity_file: str = ""


class ExoConfig(BaseModel):
    """``exo`` distributed-inference cluster lifecycle settings.

    See ``backend/exo_cli.py`` for the underlying provisioner. The
    backend's :mod:`backend.exo_provisioner` reads from here every time
    it provisions / starts / stops a node so the UI is the source of
    truth and CLI overrides only apply for one-shot manual runs.
    """

    enabled: bool = False
    # Delivery mode for the exo runtime:
    #   "prebuilt" (default) — download a notarized, prebuilt runtime
    #     artifact on demand (no git/uv/npm/rustup on the user's machine).
    #     See :mod:`backend.exo_runtime`.
    #   "source" — legacy path: clone the repo and ``uv sync`` locally.
    #     Fallback for advanced users / unsupported architectures.
    mode: str = "prebuilt"  # prebuilt | source
    # Custom prebuilt artifact source (Settings → Cluster → Advanced).
    # Leave blank to use the default GitHub releases manifest.
    # Accepts:
    #   ""                  → default manifest (GitHub releases)
    #   "https://.../exo-runtime-manifest.json"  → custom manifest URL
    #   "https://.../exo-runtime-aarch64.tar.gz" → direct artifact URL
    #   "file:///path/to/exo-runtime.tar.gz"      → local file (testing)
    prebuilt_url: str = ""
    repo_url: str = "https://github.com/exo-explore/exo.git"
    # Pins both the source-mode checkout and the prebuilt artifact lookup.
    repo_ref: str = "v1.0.71"
    api_port: int = 52415
    libp2p_port: int = 0  # 0 = OS-assigned
    base_url: str = ""    # blank ⇒ derived from api_port
    # Default model id served by the cluster when ``LLMConfig.provider``
    # is ``"exo"``. Must match an id from ``GET /v1/models`` on the
    # cluster.  Empty until the user picks one in
    # Settings → LLM → Models.
    model_name: str = ""
    auto_start: bool = False
    auto_provision: bool = True
    # macOS only.  When the desktop binary holds the Local-Network grant
    # (signed Otto.app), set this so we skip the osascript→Terminal
    # workaround and launch exo directly under the backend process.
    no_terminal_wrap: bool = False
    # Minimum number of cluster nodes that must hold a shard of a placed
    # instance.  exo's ``POST /place_instance`` defaults this to 1, which
    # tells the scheduler "one device is fine if the model fits".  Bump to
    # 2+ to force a pipeline-parallel split across multiple macs even when
    # the model fits on one node — useful for very long contexts (KV cache
    # gets distributed too) or when you want both nodes visibly engaged.
    # Note: cross-node tensor traffic costs tokens/sec, so the right value
    # depends on whether you're optimising for capacity or latency.
    min_nodes: int = 1
    remotes: list[ExoRemoteConfig] = Field(default_factory=list)
    # Default path used by the Cluster setup wizard when generating a
    # dedicated ED25519 keypair for cluster bootstrap. ``~`` is expanded
    # at use site. Empty falls back to ``~/.ssh/id_ed25519_exo``.
    default_keypair_path: str = "~/.ssh/id_ed25519_exo"

    # ── Generation knobs (sent per request by model_factory.create_llm) ──
    #
    # Maximum tokens generated per response, sent as ``max_tokens``.  Without
    # it exo falls back to the model's full context window, letting a model
    # that fails to emit a stop token run away to a multi-minute completion.
    # 8192 keeps responses bounded while leaving ample room for tool-calling
    # turns (mirrors the oMLX default).
    max_tokens: int = 8192
    # Chain-of-thought reasoning.  Sent per request as ``enable_thinking`` (exo
    # maps it to ``reasoning_effort``).  Defaults OFF: thinking can add
    # thousands of tokens per turn on Qwen3 / DeepSeek / GLM models, so
    # disabling it is the single biggest latency win for an interactive agent.
    enable_thinking: bool = False

    # ── Placement knobs (sent to POST /place_instance) ───────────────────
    #
    # Sharding strategy when a model is split across nodes:
    #   "Pipeline" (default) — layers split across devices; best single-request
    #     latency (what an agent doing one turn at a time wants) and works for
    #     every model.
    #   "Tensor" — each layer split across devices; up to 1.8x (2 nodes) /
    #     3.2x (4 nodes) *throughput* for concurrent requests, but only models
    #     with a tensor sharding strategy support it.
    # No effect on a single-node placement (min_nodes=1, model fits one mac).
    sharding: str = "Pipeline"  # Pipeline | Tensor
    # Instance transport / collective backend:
    #   "MlxRing" (default) — ring all-reduce; works over any network, the
    #     universally compatible choice.
    #   "MlxJaccl" — JACCL backend; lower latency but requires Thunderbolt 5 /
    #     RDMA-capable hardware (M4 Pro/Max, macOS 26.2+).  Only flip this on a
    #     qualified Thunderbolt cluster.
    instance_meta: str = "MlxRing"  # MlxRing | MlxJaccl

    @property
    def effective_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        return f"http://127.0.0.1:{self.api_port}"


class OmlxConfig(BaseModel):
    """``oMLX`` local inference server lifecycle settings.

    oMLX (https://github.com/jundot/omlx) is an Apple-Silicon LLM
    inference server with continuous batching, paged KV cache, and an
    OpenAI-compatible API.  Otto can use it as the ``omlx`` LLM
    provider — ``model_factory.create_llm`` builds a ``ChatOpenAI``
    pointed at ``effective_base_url + /v1``.

    The provisioner (:mod:`backend.omlx_provisioner`) does NOT bundle
    oMLX.  It detects whether the user already installed it (via
    Homebrew, the ``.app`` bundle, or a custom path), and installs it
    from the setup flow when needed.  Install is brew-optional: it
    prefers ``brew tap jundot/omlx && brew install omlx`` when Homebrew
    is present, and otherwise downloads the official GitHub ``.dmg`` /
    release (whose bundle ships a CLI launcher) — no admin or Homebrew
    required.  Lifecycle uses ``brew services start omlx`` when
    available, with a manual ``omlx serve`` fallback so users without
    Homebrew can still run a managed daemon.

    Mirrors the shape of :class:`ExoConfig` so the wizard / settings
    flows reuse the same patterns.
    """

    enabled: bool = False
    # Port 8000 was the original default but it conflicts with common dev
    # servers (FastAPI, Flask, etc.) causing false-positive reachability
    # checks. 52414 is in the same private range as the EXO default (52415)
    # but dedicated to oMLX to avoid cross-service confusion.
    # Existing installs with a non-empty base_url are unaffected.
    api_port: int = 52414
    base_url: str = ""    # blank ⇒ derived from api_port
    # Default model id served by the oMLX server when ``LLMConfig.provider``
    # is ``"omlx"``. Must match an id from ``GET /v1/models`` on the
    # running server. Empty until the user picks one in
    # Settings → LLM → Models.
    model_name: str = ""
    auto_start: bool = False
    # Homebrew install knobs.  Values are exposed in Settings so a user
    # can pin to a non-default tap (e.g. for a custom build) without
    # editing config.json by hand.
    brew_tap: str = "jundot/omlx"
    brew_tap_url: str = "https://github.com/jundot/omlx"
    brew_formula: str = "omlx"
    # Optional explicit binary path. Empty ⇒ resolve via PATH lookup.
    cli_path: str = ""

    # Directories oMLX scans to discover local models.  Passed to the
    # oMLX admin API as ``model_dirs`` during provisioning.  Tilde (~)
    # is expanded at provision time.  Defaults to the HuggingFace hub
    # cache so models downloaded via ``huggingface-cli`` or the HF Python
    # library are auto-discovered without any manual configuration.
    model_dirs: list[str] = Field(
        default_factory=lambda: ["~/.cache/huggingface/hub"],
    )

    # Admin API key Otto uses to drive oMLX's /admin/api/* endpoints
    # (provisioning model_dirs, reloading, downloading, etc).  Generated
    # once and persisted; never returned in API responses unmasked.
    # Empty ⇒ provisioner generates one on first run.
    admin_api_key: str = ""

    # Maximum context length (tokens) applied to the running server via
    # sampling_max_context_window in /admin/api/global-settings.  oMLX
    # defaults to 32768; set higher to handle longer conversations.
    max_context_window: int = 131072

    # Enable chain-of-thought / reasoning output.  oMLX has no global
    # admin toggle for thinking, so this is applied per request as
    # ``chat_template_kwargs={"enable_thinking": <bool>}`` by
    # ``model_factory.create_llm``.  Only affects models whose chat
    # template supports a thinking switch (e.g. Qwen3).
    thinking_enabled: bool = False

    # Maximum tokens generated per response, sent per request as
    # ``max_tokens`` by ``model_factory.create_llm``.  Without this the
    # server falls back to ``sampling.max_tokens`` (32768 by default),
    # which lets a model that fails to emit a stop token run away to the
    # full cap — a single ~10-minute completion.  8192 keeps responses
    # bounded while leaving ample room for tool-calling turns.
    max_tokens: int = 8192

    @property
    def effective_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        return f"http://127.0.0.1:{self.api_port}"


# Embedding models that require HuggingFace authentication or are gated.
# When a persisted config.json references one of these, it is silently
# migrated to the default public model on load.
_GATED_EMBEDDING_MODELS: frozenset[str] = frozenset({
    "mlx-community/nomic-embed-text-v1.5-mlx",
    "nomic-ai/nomic-embed-text-v1.5",
})


class EmbeddingConfig(BaseModel):
    """Semantic embedding index (sub-feature of memory).

    Only active when the parent ``MemoryConfig.enabled`` is ``True``.
    The embedder uses ``sentence-transformers`` and works on any platform.
    """

    enabled: bool = True
    # HuggingFace repo id for the embedding model.  all-MiniLM-L6-v2 is
    # 90 MB, 384-dim, completely public (no HF token required), and works
    # with mlx-embeddings' BERT backend on Apple Silicon.
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Maximum chars per chunk fed to the embedder (reuses the doc_researcher
    # splitter so all indexed content is at the same granularity).
    chunk_size: int = 1500
    chunk_overlap: int = 150

    @field_validator("model_name", mode="before")
    @classmethod
    def _migrate_gated_model(cls, v: object) -> object:
        if isinstance(v, str) and v in _GATED_EMBEDDING_MODELS:
            return "sentence-transformers/all-MiniLM-L6-v2"
        return v


class MemoryConfig(BaseModel):
    """Memory consolidation settings."""

    enabled: bool = False
    # Legacy combined flag — kept for backward compatibility with older
    # config.json files.  When set it implicitly enables BOTH of the newer
    # per-layer toggles below so existing users keep their current behavior.
    inject_enabled: bool = False
    # Layer 1: inject MEMORY.md into the system prompt once at session build.
    # Zero extra LLM calls per turn; stable, predictable context.
    inject_on_session_start: bool = False
    # Layer 2: run a per-turn relevance ranker (Haiku / MLX) and inject the
    # top-K matching topic files.  Adaptive but costs one side-query per turn.
    inject_realtime: bool = False
    model_name: str = ""
    # LLM stack used for the consolidation + per-turn ranking side-queries.
    # ``follow_main`` reuses whatever the main chat is on (Anthropic Haiku
    # auto-pick when frontier, MLX text model otherwise).  ``frontier`` forces
    # Anthropic / Bedrock and uses ``model_name`` (or auto-Haiku).  ``mlx``
    # forces a local MLX model and uses ``mlx_model`` (or the global MLX text
    # model when blank).
    llm_family: str = "follow_main"  # follow_main | frontier | mlx
    mlx_model: str = ""  # HF repo id when llm_family == "mlx"
    min_hours: int = 24
    min_sessions: int = 5
    retention_days: int = 30
    max_memory_files: int = 200
    max_index_kb: int = 25
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    @property
    def embedding_enabled(self) -> bool:
        """True only when both memory and the embedding sub-feature are on."""
        return self.enabled and self.embedding.enabled

    @property
    def effective_inject_on_session_start(self) -> bool:
        """Layer 1 on, honoring the legacy ``inject_enabled`` flag."""
        return self.inject_on_session_start or self.inject_enabled

    @property
    def effective_inject_realtime(self) -> bool:
        """Layer 2 on, honoring the legacy ``inject_enabled`` flag."""
        return self.inject_realtime or self.inject_enabled


class ActivityConfig(BaseModel):
    """User activity timeline tracker.

    When enabled, the backend periodically polls macOS for the foreground
    app, window title, browser URL, and active document path, then stores
    a compact row in a local SQLite DB.  No screenshots, no pixel capture
    — only metadata.  Searchable via FTS5 from both the UI and the agent.

    The tracker is opt-in (off by default) and runs entirely on-device.
    """

    enabled: bool = False
    # How often to poll for the active window (seconds).  Lower = finer
    # granularity, higher CPU cost.  Sensible range: 5-60 seconds.
    interval_secs: int = 15
    # How many days of activity to retain.  Older rows are pruned by a
    # daily cleanup pass.  0 = keep forever.
    retain_days: int = 30
    # Apps that should never be recorded (case-insensitive substring
    # match against the app's localized name).  Defaults exclude common
    # password / banking apps for safety.
    exclude_apps: list[str] = Field(
        default_factory=lambda: ["1Password", "Bitwarden", "Banking"],
    )
    # Skip ticks when no keyboard/mouse input has happened for this many
    # seconds.  Prevents recording an 8-hour overnight span where the
    # user wasn't actually present.  0 disables idle detection.
    idle_threshold_secs: int = 180
    # Drop spans shorter than this when transitioning to a new window.
    # Cleans up tab-flicking noise without losing the long-running
    # spans the user actually cared about.  0 disables.
    min_span_secs: int = 3
    # Force a new row after a span on the same window has been running
    # for this long.  Without this, focused 90-minute Cursor sessions
    # collapse into a single row that loses everything that happened
    # inside.  10 minutes gives ~6 rows/hour during deep focus, each
    # preserving its own context snapshot.  0 disables sharding.
    max_span_secs: int = 600
    # Max characters retained in a row's rolling ``context`` log.  Each
    # tick's selection / focused-input is appended (or merged with the
    # previous entry if the user is just typing more characters).  When
    # the buffer overflows, the oldest entries are trimmed off the
    # front.  0 disables the rolling log (revert to overwrite-only).
    context_max_chars: int = 4096
    # Max characters captured from the focused UI element's ``AXValue``
    # attribute each tick.  Native text apps (Notes, Mail, Pages, Xcode)
    # expose their full document content here.  8000 chars covers a
    # typical page of writing; raise it for longer documents.
    # 0 disables AXValue capture entirely.
    field_val_max_chars: int = 8000
    # Maximum SQLite file size in megabytes.  When the DB exceeds this
    # limit during the hourly cleanup pass, the oldest rows are deleted
    # in batches until the file is back under the cap (the file is also
    # vacuumed so the freed pages are returned to the OS).  0 = unlimited.
    max_db_mb: int = 5120
    # Phase 0: pull the visible page text from the active browser tab
    # via AppleScript ``do JavaScript`` / ``execute javascript`` so the
    # row's context contains what the user is actually reading, not
    # just the URL + title.  Capped at this many chars.  0 disables.
    # Requires "Allow JavaScript from Apple Events" in Safari (Develop
    # menu) and Chrome/Brave (View → Developer).  Arc allows it by
    # default.  Soft-fails on permission errors.
    browser_text_max_chars: int = 4000
    # Phase 0: walk the Accessibility tree of the focused window and
    # collect text from static-text / value-bearing elements.  Captures
    # rich context for Electron apps (Cursor, VS Code, Slack, Discord,
    # Notion) and AppKit apps without a single dominant AXValue.
    # 0 disables the tree walk.
    ax_walk_max_chars: int = 3000
    # Maximum recursion depth when descending the AX tree.  Electron
    # apps (Cursor, Slack, Discord, VS Code, Notion) nest text inside
    # 15-25 layers of empty Chromium <div> AXGroups before reaching
    # the actual AXStaticText leaves; native AppKit apps are much
    # shallower (~5-8 levels).  25 is the sweet spot — costs ~30ms
    # extra per Electron tick but actually reaches message bodies.
    ax_walk_max_depth: int = 25


class PrivacyConfig(BaseModel):
    """Verifiable airplane-mode configuration.

    When ``enabled`` is True, Otto refuses to construct any LLM that
    would send data off-device.  Only the providers in
    :attr:`local_only_providers` (MLX, oMLX, exo) are allowed.

    The app-layer guard is the actual security boundary: it lives
    inside the Python process and is checked at session start, on every
    settings update, and on every tool call that would dial out.

    The optional macOS ``pf`` (packet filter) anchor rendered by
    :func:`backend.privacy_lock.render_pf_template` is a defence-in-
    depth layer the user can install with ``sudo`` to make the
    network block enforceable by the kernel rather than the
    application.  It is NOT auto-installed; doing so would require
    silent ``sudo`` and we deliberately keep that decision in the
    user's hands.
    """

    enabled: bool = False
    # Providers that may continue to run while privacy mode is engaged.
    # These all run weights on-device or on a local network the user
    # explicitly enrolled (exo cluster).
    local_only_providers: list[str] = Field(
        default_factory=lambda: ["mlx", "omlx", "exo"],
    )
    # Host:port pairs (e.g. exo remote nodes, internal mirrors) that
    # are allowed to receive traffic when the kernel-level pf rules
    # are installed.  The app-layer guard ignores this field; it
    # exists solely so the rendered pf template knows what to exempt
    # from the egress block.
    allowed_hosts: list[str] = Field(default_factory=list)
    # Whether to keep loopback (127.0.0.1, ::1) reachable in the pf
    # template.  Almost always wanted; without it the backend can't
    # talk to its own MCP subprocesses.
    allow_loopback: bool = True
    # Whether to keep mDNS / Bonjour reachable so the exo cluster
    # can announce nodes on the LAN.
    allow_mdns: bool = True
    # Stable name for the pf anchor, used both for ``pfctl -a`` and
    # for status parsing.  Changing this after engagement breaks
    # ``pfctl -F -a <anchor>``, so it's deliberately a config knob
    # rather than a hard-coded constant.
    pf_anchor: str = "otto.privacy"
    # Wall-clock ISO timestamp of the most recent engagement.  Empty
    # when not engaged.  Surfaced in audit logs and the status panel.
    engaged_at: str = ""
    # Random hex marker stamped at engagement time so a single audit
    # log can be matched to one specific engage/disengage cycle.
    audit_token: str = ""


class AmbientConfig(BaseModel):
    """Ambient assistant — proactive, out-of-chat hints.

    When enabled, a lightweight background agent periodically analyses
    memory, session history, and macOS activity to surface concrete,
    actionable suggestions.  Each hint can be opened in a pre-filled
    chat or approved to run as a background session (still gated by
    the normal HITL flow for any risky tool calls).

    Entirely opt-in and off by default so it never interrupts users
    who haven't explicitly turned it on.
    """

    enabled: bool = False

    # ── Own smaller model (mirrors MemoryConfig pattern) ─────────────────
    # Defaults to "follow_main" so the ambient sweep uses whatever the main
    # chat provider is.  This avoids silently loading a *second* in-process
    # MLX model (and a fresh warmup allocation) into unified Metal memory
    # while the user has deliberately switched the chat to a cloud/frontier
    # provider — a combination that previously caused GPU OOM crashes.  Set
    # this to "mlx" explicitly to run sweeps on a small local model.
    llm_family: str = "follow_main"  # follow_main | frontier | mlx | exo
    # HF repo id used when llm_family == "mlx".  The Qwen3-1.7B model is
    # 1.1 GB and already downloaded by the chat-setup flow on Apple Silicon,
    # so the first sweep is typically free.
    mlx_model: str = "mlx-community/Qwen3-1.7B-4bit"
    # Cloud model when llm_family == "frontier"; blank → auto-pick Haiku.
    model_name: str = ""

    # ── Cadence ──────────────────────────────────────────────────────────
    interval_mins: int = 30
    # Only run when the activity tracker reports the user is idle.  Avoids
    # interrupting active work even if the interval fires mid-session.
    idle_only: bool = True
    # Also trigger a sweep shortly after each session completes (debounced).
    react_to_session_end: bool = True

    # ── Context sources ───────────────────────────────────────────────────
    use_memory: bool = True
    use_sessions: bool = True
    use_activity: bool = True
    use_history: bool = True
    # How many hours back each context gatherer looks.  Applies to sessions,
    # macOS activity, and history aggregation.  Memory is a static index so
    # it is not time-filtered.
    lookback_hours: int = 24

    # ── Quality / rate limits ─────────────────────────────────────────────
    # Confidence threshold (0–1); hints below this are silently discarded.
    min_confidence: float = 0.6
    max_hints_per_day: int = 10
    # Minimum hours before an equivalent hint on the same topic can resurface.
    cooldown_hours: int = 4
    # Quiet window: no notifications between quiet_hours_start and quiet_hours_end
    # (local 24h clock).  The sweep still runs; hints are queued until the
    # window ends rather than dropped.
    quiet_hours_start: int = 22  # 10 pm
    quiet_hours_end: int = 8     # 8 am

    # ── Approval ──────────────────────────────────────────────────────────
    # When False (default) each hint only offers "Open in chat" (pre-filled
    # draft; user must press Send).  When True, an "Approve & run" button
    # is also shown, spawning a background session immediately.
    allow_auto_run: bool = True


class VoiceConfig(BaseModel):
    """On-device voice layer (STT + optional wake word; no TTS).

    All weights are local / on-device.  The subsystem is entirely
    opt-in: ``enabled=False`` (default) means zero mic access and
    zero model loading.  When enabled, ``activation_mode`` selects
    between a push-to-talk mic button and always-on wake-word detection.
    """

    enabled: bool = False

    # ── Activation ────────────────────────────────────────────────────────
    # "ptt"      – push-to-talk (mic button held)
    # "wakeword" – continuous listening, fire on "Hey Otto"
    activation_mode: str = "ptt"  # ptt | wakeword

    # Global hotkey combo for PTT, e.g. "Control+Space".  Empty = mic
    # button in UI only (no system-wide shortcut).
    ptt_hotkey: str = ""

    # ── STT ───────────────────────────────────────────────────────────────
    stt_enabled: bool = True
    stt_model: str = "mlx-community/whisper-large-v3-turbo"
    # BCP-47 language tag passed to Whisper ("" = auto-detect)
    stt_language: str = ""

    # ── Wake word ─────────────────────────────────────────────────────────
    wake_enabled: bool = False
    wake_model: str = "hey_otto"   # bundled ONNX at backend/voice/models/hey_otto.onnx
    # Seconds of silence after the last detected word before the mic stops
    vad_silence_secs: float = 1.0

    # ── Audio device ──────────────────────────────────────────────────────
    # Empty string = system default microphone
    mic_device: str = ""


class SetupChatConfig(BaseModel):
    """Configuration for the conversational first-run setup agent.

    A small local MLX model (Qwen3-1.7B-4bit, 1.1 GB) handles the
    Q&A flow when the machine is eligible (Apple Silicon, ≥ 8 GB RAM).
    Falls back to template strings on ineligible hardware or when the
    model has not been downloaded yet.
    """

    setup_model_id: str = "mlx-community/Qwen3-1.7B-4bit"
    setup_max_tokens: int = 150
    setup_temp: float = 0.4


class SetupState(BaseModel):
    """First-run setup wizard progress.

    ``completed`` flips ``True`` only when the user clicks "Open Otto" on
    the final wizard screen.  ``dismissed`` flips ``True`` when the user
    explicitly clicks "Skip".  Either one suppresses the wizard on
    subsequent launches.  ``current_step`` lets a re-opened wizard pick
    up where it left off if neither flag is set yet.
    """

    completed: bool = False
    dismissed: bool = False
    current_step: str = "welcome"
    # Steps the user has touched (for resume hinting). Free-form list of
    # step ids — kept short, no schema validation.
    completed_steps: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=_default_mcp_servers)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    claude_hook: ClaudeHookConfig = Field(default_factory=ClaudeHookConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    openclaw: OpenClawConfig = Field(default_factory=OpenClawConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    exo: ExoConfig = Field(default_factory=ExoConfig)
    omlx: OmlxConfig = Field(default_factory=OmlxConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    setup: SetupState = Field(default_factory=SetupState)
    setup_chat: SetupChatConfig = Field(default_factory=SetupChatConfig)
    # When True, HITL approval prompts are automatically answered with
    # "approve" — except for commands flagged as high-risk by the safety
    # middleware, which always require manual confirmation.
    auto_approve_commands: bool = False

    # In-chat nudge: after each session Otto checks whether the task looks
    # repeatable and offers an inline chip to create a schedule/trigger.
    ambient_suggest_recurrence: bool = False
    # Ambient assistant: proactive out-of-chat hints driven by memory,
    # sessions, and macOS activity.  See AmbientConfig for all controls.
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)

    @model_validator(mode="after")
    def _fix_memory_llm_family(self) -> "AppConfig":
        """Auto-correct memory.llm_family when it would silently break.

        If memory consolidation is set to use MLX (``llm_family == "mlx"``)
        but no specific MLX model is pinned (``mlx_model`` is empty) AND the
        main provider is not a local one, fall back to ``"follow_main"``.

        This prevents a broken state where consolidation inherits the global
        MLX model (e.g. a partially-downloaded Qwen3-14B) while the user is
        running on Anthropic — causing every consolidation run to fail with
        missing-weight errors.
        """
        _LOCAL_PROVIDERS = {"mlx", "omlx", "exo"}
        mem = self.memory
        if (
            mem.llm_family == "mlx"
            and not (mem.mlx_model or "").strip()
            and (self.llm.provider or "").lower() not in _LOCAL_PROVIDERS
        ):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[config] memory.llm_family='mlx' with no mlx_model and provider='%s' "
                "— upgrading to 'follow_main' to avoid broken consolidation runs.",
                self.llm.provider,
            )
            mem.llm_family = "follow_main"
        return self

    @model_validator(mode="after")
    def _migrate_wake_model(self) -> "AppConfig":
        """Migrate legacy openWakeWord model names to the bundled hey_otto.

        Earlier builds defaulted ``wake_model`` to ``hey_jarvis_v0.1`` (or
        other openWakeWord built-ins).  "Hey Otto" is now the only supported
        wake phrase, so any legacy value is silently upgraded.
        """
        _LEGACY_WAKE_MODELS = {"hey_jarvis_v0.1", "alexa_v0.1", "hey_mycroft_v0.1"}
        if self.voice.wake_model in _LEGACY_WAKE_MODELS:
            self.voice.wake_model = "hey_otto"
        return self

    @classmethod
    def load(cls) -> AppConfig:
        config_path = get_app_data_dir() / "config.json"
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            cfg = cls.model_validate(data)
            # Overlay keychain secrets (and migrate any legacy plaintext)
            # BEFORE _ensure_default_servers, which may itself call save():
            # saving with un-hydrated (empty) secrets would wipe the vault.
            cfg._hydrate_secrets_from_vault(data)
            cfg._ensure_default_servers()
            return cfg
        return cls()

    @classmethod
    async def aload(cls) -> AppConfig:
        """Non-blocking variant of :meth:`load` for use in async contexts."""
        return await asyncio.to_thread(cls.load)

    _RENAMED_SERVER_IDS: dict[str, str] = {
        "deepeval": "agent-eval-service",
        "claude-eval-service": "agent-eval-service",
        "transcript-reader": "claude-eval-hook",
        "agent-eval-hook": "claude-eval-hook",
        "claude-transcript-reader": "claude-eval-hook",
    }

    def _ensure_default_servers(self) -> None:
        """Inject missing builtin MCP servers and sync their name/builtin flag.

        Non-builtin defaults are only created on first run (no config.json).
        Stale entries from renamed server IDs are removed automatically.

        For source-bundled built-in MCPs (registered via
        :mod:`backend.builtin_mcps`), this also:

        * Mirrors the repo-resident ``server.py`` / ``requirements.txt``
          into ``<app_data>/mcp_server/<id>/`` so a fresh install has
          working files before any connection attempt.
        * Refreshes ``command`` / ``args`` on existing entries (handles
          the case where the same id was previously registered as an
          agent-generated MCP and we now own it as a built-in).
        """
        stale_ids = set(self._RENAMED_SERVER_IDS.keys())
        self.mcp_servers = [s for s in self.mcp_servers if s.id not in stale_ids]

        try:
            from backend.builtin_mcps import sync_builtin_mcp_files
            sync_builtin_mcp_files()
        except Exception as exc:
            logger.warning(
                "Failed to sync built-in MCP source files: %s", exc,
            )

        existing = {s.id: s for s in self.mcp_servers}
        dirty = False
        for default_srv in _default_mcp_servers():
            if not default_srv.builtin:
                continue
            if default_srv.id not in existing:
                self.mcp_servers.append(default_srv)
                dirty = True
            else:
                cur = existing[default_srv.id]
                if cur.name != default_srv.name:
                    cur.name = default_srv.name
                    dirty = True
                if cur.builtin != default_srv.builtin:
                    cur.builtin = default_srv.builtin
                    dirty = True
                # Stdio built-ins live at deterministic paths under
                # ``mcp_server/<id>/``; if the user previously had this
                # id as an agent-generated MCP its command/args might
                # still point at a stale location.  Re-pin to the
                # built-in's canonical paths.
                if default_srv.transport == "stdio":
                    if cur.command != default_srv.command:
                        cur.command = default_srv.command
                        dirty = True
                    if list(cur.args or []) != list(default_srv.args or []):
                        cur.args = list(default_srv.args or [])
                        dirty = True
                    if cur.transport != "stdio":
                        cur.transport = "stdio"
                        dirty = True
                    if cur.generated:
                        cur.generated = False
                        dirty = True
                    if list(cur.required_secrets or []) != list(
                        default_srv.required_secrets or []
                    ):
                        cur.required_secrets = list(
                            default_srv.required_secrets or []
                        )
                        dirty = True
                    if list(cur.optional_secrets or []) != list(
                        default_srv.optional_secrets or []
                    ):
                        cur.optional_secrets = list(
                            default_srv.optional_secrets or []
                        )
                        dirty = True
                    if cur.requires_os != default_srv.requires_os:
                        cur.requires_os = default_srv.requires_os
                        dirty = True

        ch_srv = existing.get("claude-eval-hook")
        if ch_srv is not None and ch_srv.enabled != self.claude_hook.enabled:
            ch_srv.enabled = self.claude_hook.enabled
            dirty = True

        oc_srv = existing.get("openclaw-eval-hook")
        if oc_srv is not None and oc_srv.enabled != self.openclaw.enabled:
            oc_srv.enabled = self.openclaw.enabled
            dirty = True

        if platform_label() != "macos":
            n_before = len(self.mcp_servers)
            self.mcp_servers = [
                s for s in self.mcp_servers
                if s.id not in ("macos-native", "macos-osascript")
            ]
            if len(self.mcp_servers) != n_before:
                dirty = True

        if dirty:
            self.save()

    def save(self) -> None:
        config_dir = get_app_data_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        data = self.model_dump(mode="json")
        # Route secret fields into the keychain and blank them on disk.
        # ``self`` is never mutated, so the in-memory model keeps real
        # values for apply_to_environ() after a save.
        self._route_secrets_to_vault(data)
        config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _route_secrets_to_vault(self, data: dict) -> None:
        """Move secret values out of *data* (the on-disk dict) into the
        OS keychain, blanking each field in *data*.

        All secrets are stored as a single consolidated keychain entry
        (one JSON bundle) rather than one row per field.  macOS authorises
        keychain access per item, so a single write means at most one
        authorisation prompt on first-run setup instead of one per secret.

        When the keychain is unavailable (headless Linux/CI, no Secret
        Service) the values are left in *data* so the app keeps working,
        falling back to the legacy plaintext behaviour.  If the vault
        write fails, the affected fields are left in place rather than
        silently dropped.
        """
        if not app_vault.available():
            _warn_vault_unavailable_once()
            return
        bundle: dict[str, str] = {}
        for path, account in _SECRET_FIELDS:
            value = _dict_get_path(data, path)
            if isinstance(value, str) and value:
                bundle[account] = value
        try:
            if bundle:
                app_vault.set_bundle(bundle)
            else:
                app_vault.delete_bundle()
        except CredentialVaultError as exc:
            # Leave the secret fields on disk so the app keeps working;
            # don't blank values we failed to persist to the keychain.
            logger.warning("vault: failed to store app secret bundle: %s", exc)
            return
        for path, _account in _SECRET_FIELDS:
            _dict_set_path(data, path, "")

    def _hydrate_secrets_from_vault(self, disk_data: dict) -> None:
        """Overlay keychain-stored secrets onto the in-memory model and
        migrate older storage formats forward.

        Secrets now live in a single consolidated keychain bundle.  Two
        legacy formats are migrated transparently:

        * **Per-field keychain rows** written by older versions (one
          ``otto.app`` entry per secret).  Read as a fallback when the
          bundle is absent.  The stale rows are deliberately **not**
          deleted — deleting them re-triggers the per-item keychain
          prompt this consolidation exists to avoid, and they're harmless
          dead entries once the bundle is written.
        * **Plaintext secrets on disk** from before the vault existed.

        A re-save (which writes the bundle and scrubs disk) runs only
        when one of those legacy formats is detected, so steady-state
        loads stay read-only.
        """
        if not app_vault.available():
            return
        try:
            bundle = app_vault.get_bundle()
        except CredentialVaultError:
            bundle = None

        needs_migration = False
        legacy_fields = bundle is None  # fall back to per-field rows
        for path, account in _SECRET_FIELDS:
            stored: Optional[str] = None
            if bundle is not None:
                stored = bundle.get(account) or None
            elif legacy_fields:
                try:
                    stored = app_vault.get(account)
                except CredentialVaultError:
                    stored = None
                if stored:
                    # Found a legacy per-field row — fold it into a bundle.
                    needs_migration = True
            disk_val = _dict_get_path(disk_data, path)
            if isinstance(disk_val, str) and disk_val:
                # Legacy plaintext value on disk — schedule migration.
                needs_migration = True
                if not stored:
                    stored = disk_val
            if stored:
                _model_set_path(self, path, stored)
        if needs_migration:
            # Pushes secrets to the consolidated bundle and scrubs disk.
            self.save()

    async def asave(self) -> None:
        """Non-blocking variant of :meth:`save` for use in async contexts."""
        await asyncio.to_thread(self.save)

    def is_first_run(self) -> bool:
        """Whether the setup wizard should be shown.

        Returns ``True`` when no config exists yet OR when the user has
        neither completed nor explicitly dismissed the wizard.  This
        survives mid-wizard saves: each step writes the partial config,
        but the wizard keeps showing until the user clicks Finish or
        Skip on the last screen.
        """
        config_path = get_app_data_dir() / "config.json"
        if not config_path.exists():
            return True
        return not (self.setup.completed or self.setup.dismissed)

    # ------------------------------------------------------------------
    # Backward-compat: populate os.environ so the existing Environment
    # class and model_factory work without changes.
    # ------------------------------------------------------------------

    def to_env_dict(self) -> dict[str, str]:
        env: dict[str, str] = {}
        env["LLM_PROVIDER"] = self.llm.provider

        a = self.llm.anthropic
        o = self.llm.openai
        m = self.llm.mlx

        def _anthropic_block() -> dict[str, str]:
            return {
                "ANTHROPIC_API_KEY": a.api_key,
                "ANTHROPIC_MODEL_NAME": a.model_name,
                "ANTHROPIC_MODEL_PROVIDER": a.model_provider,
                "ANTHROPIC_BEDROCK_REGION": a.bedrock_region,
                "ANTHROPIC_BEDROCK_AUTH_MODE": a.bedrock_auth_mode,
                "ANTHROPIC_MAX_TOKENS": str(a.max_tokens),
                "ANTHROPIC_THINKING_FLAG": str(a.thinking_enabled).lower(),
                "ANTHROPIC_THINKING_BUDGET": str(a.thinking_budget),
                "ANTHROPIC_TOOL_EFFICIENT_FLAG": str(a.tool_efficient).lower(),
            }

        def _openai_block() -> dict[str, str]:
            # Write the active provider's key to OPENAI_API_KEY so the
            # model factory always reads from the same env var.
            active_key = o.azure_api_key if o.model_provider == "azure" else o.api_key
            return {
                "OPENAI_API_KEY": active_key,
                "OPENAI_MODEL_NAME": o.model_name,
                "OPENAI_MODEL_PROVIDER": o.model_provider,
                "OPENAI_AZURE_ENDPOINT": o.azure_endpoint,
                "OPENAI_AZURE_API_VERSION": o.azure_api_version,
                "OPENAI_AZURE_DEPLOYMENT": o.azure_deployment,
                "OPENAI_MAX_TOKENS": str(o.max_tokens),
                "OPENAI_TEMPERATURE": str(o.temperature),
            }

        def _mlx_block() -> dict[str, str]:
            return {
                "HF_LLM_MODEL_ID": m.hf_llm_model_id,
                "HF_VLM_MODEL_ID": m.hf_vlm_model_id,
                "HF_DRAFT_LLM_MODEL_ID": m.hf_draft_llm_model_id,
                "HF_TOKEN": m.hf_token,
                "HF_HUB_CACHE": resolve_hf_hub_cache_dir(m.hf_hub_cache),
                "MLX_MAX_TOKENS": str(m.mlx_max_tokens),
                "MLX_TEMP": str(m.mlx_temp),
                "MLX_VERBOSE": str(m.mlx_verbose).lower(),
                "MLX_THINKING": str(m.mlx_thinking).lower(),
                "MLX_PROMPT_CACHE": str(m.mlx_prompt_cache).lower(),
                "MLX_SYSTEM_PROMPT_CACHE": str(m.mlx_system_prompt_cache).lower(),
                "MLX_REPETITION_PENALTY": str(m.mlx_repetition_penalty),
                "MLX_KV_BITS": str(m.mlx_kv_bits) if m.mlx_kv_bits is not None else "",
                "MLX_KV_GROUP_SIZE": str(m.mlx_kv_group_size),
                "MLX_NUM_DRAFT_TOKENS": str(m.mlx_num_draft_tokens),
                "MLX_PROMPT_CACHE_MAX_TOKENS": str(m.mlx_prompt_cache_max_tokens),
                "MLX_TURBO_LEVEL": (m.turbo_level or "off").strip().lower(),
                "MLX_TURBO_SSD_DIR": m.turbo_ssd_dir,
                "MLX_TURBO_SSD_MAX_GB": str(m.turbo_ssd_max_gb),
                "MLX_TURBO_TQ_BITS": str(m.turbo_tq_bits),
                "MLX_TURBO_BLOCK_SIZE": str(m.turbo_block_size),
            }

        anthropic_keys = tuple(_anthropic_block().keys())
        openai_keys = tuple(_openai_block().keys())
        mlx_keys = tuple(_mlx_block().keys())

        def _clear_anthropic() -> None:
            for k in anthropic_keys:
                env[k] = ""

        def _clear_openai() -> None:
            for k in openai_keys:
                env[k] = ""

        def _clear_mlx() -> None:
            for k in mlx_keys:
                env[k] = ""

        use_mlx_hub = bool(m.hf_llm_model_id.strip())
        want_anthropic = bool(a.api_key.strip()) or a.model_provider in ("bedrock", "anthropic_bedrock")  # latter: legacy spelling
        want_openai = bool(o.api_key.strip()) or o.model_provider == "azure"

        if self.llm.provider == "mlx":
            env.update(_mlx_block())
            if want_anthropic:
                env.update(_anthropic_block())
            else:
                _clear_anthropic()
            if want_openai:
                env.update(_openai_block())
            else:
                _clear_openai()
        elif self.llm.provider == "openai":
            env.update(_openai_block())
            if want_anthropic:
                env.update(_anthropic_block())
            else:
                _clear_anthropic()
            if use_mlx_hub:
                env.update(_mlx_block())
            else:
                env["HF_LLM_MODEL_ID"] = ""
                env["HF_VLM_MODEL_ID"] = ""
                env["HF_DRAFT_LLM_MODEL_ID"] = ""
                env["HF_TOKEN"] = ""
                env["HF_HUB_CACHE"] = ""
                _clear_mlx()
        elif self.llm.provider in ("omlx", "exo", "cohere"):
            # Local-server and non-Anthropic cloud providers.  Only populate
            # the Anthropic block when the user has valid Anthropic credentials
            # (e.g. for memory consolidation with llm_family="frontier" or an
            # explicitly "frontier" orchestrator stack).  Setting ANTHROPIC_*
            # env vars unconditionally — even with an empty api_key — causes
            # the deepagents AnthropicPromptCachingMiddleware to instantiate a
            # live ChatAnthropic object, which immediately fails with an auth
            # error when no key is present.
            if want_anthropic:
                env.update(_anthropic_block())
            else:
                _clear_anthropic()
            if want_openai:
                env.update(_openai_block())
            else:
                _clear_openai()
            if use_mlx_hub:
                env.update(_mlx_block())
            else:
                env["HF_LLM_MODEL_ID"] = ""
                env["HF_VLM_MODEL_ID"] = ""
                env["HF_DRAFT_LLM_MODEL_ID"] = ""
                env["HF_TOKEN"] = ""
                env["HF_HUB_CACHE"] = ""
                _clear_mlx()
        else:
            # Anthropic (direct API or Bedrock) — always populate.
            env.update(_anthropic_block())
            if want_openai:
                env.update(_openai_block())
            else:
                _clear_openai()
            if use_mlx_hub:
                env.update(_mlx_block())
            else:
                env["HF_LLM_MODEL_ID"] = ""
                env["HF_VLM_MODEL_ID"] = ""
                env["HF_DRAFT_LLM_MODEL_ID"] = ""
                env["HF_TOKEN"] = ""
                env["HF_HUB_CACHE"] = ""
                _clear_mlx()

        # Deep Agent orchestrator (optional separate stack)
        orch = self.orchestrator
        po = (orch.provider_override or "").strip().lower() if orch.provider_override else ""
        if po:
            env["DEEP_AGENT_LLM_PROVIDER"] = po
            if po == "mlx":
                mid = (orch.mlx_model or "").strip()
                env["DEEP_AGENT_MLX_MODEL_ID"] = mid
                env["DEEP_AGENT_MLX_MODEL_TYPE"] = (orch.mlx_model_type or "llm").strip().lower()
            else:
                env["DEEP_AGENT_MLX_MODEL_ID"] = ""
                env["DEEP_AGENT_MLX_MODEL_TYPE"] = ""
        else:
            fam = (orch.llm_family or "follow_main").strip().lower()
            if fam not in ("follow_main", "frontier", "mlx", "exo", "openai"):
                fam = "follow_main"
            if fam == "follow_main":
                env["DEEP_AGENT_LLM_PROVIDER"] = ""
                env["DEEP_AGENT_MLX_MODEL_ID"] = ""
                env["DEEP_AGENT_MLX_MODEL_TYPE"] = ""
            elif fam == "frontier":
                env["DEEP_AGENT_LLM_PROVIDER"] = "anthropic"
                env["DEEP_AGENT_MLX_MODEL_ID"] = ""
                env["DEEP_AGENT_MLX_MODEL_TYPE"] = ""
            elif fam == "openai":
                env["DEEP_AGENT_LLM_PROVIDER"] = "openai"
                env["DEEP_AGENT_MLX_MODEL_ID"] = ""
                env["DEEP_AGENT_MLX_MODEL_TYPE"] = ""
            elif fam == "exo":
                # Reuse the cluster's configured base URL + model id.  No
                # MLX_* overrides — exo is OpenAI-compatible, not MLX-direct.
                env["DEEP_AGENT_LLM_PROVIDER"] = "exo"
                env["DEEP_AGENT_MLX_MODEL_ID"] = ""
                env["DEEP_AGENT_MLX_MODEL_TYPE"] = ""
            else:  # mlx
                env["DEEP_AGENT_LLM_PROVIDER"] = "mlx"
                env["DEEP_AGENT_MLX_MODEL_ID"] = (orch.mlx_model or "").strip()
                env["DEEP_AGENT_MLX_MODEL_TYPE"] = (orch.mlx_model_type or "llm").strip().lower()

        # Orchestrator prompt size mode.  Resolved at session-build time
        # in ``deep_agent.prompt.build_orchestrator_prompt`` via
        # ``Environment.use_lite_orchestrator_prompt()``.
        pm = (orch.prompt_mode or "auto").strip().lower()
        env["LOCAL_PROMPT_MODE"] = pm if pm in ("auto", "full", "lite") else "auto"

        # Universal recursion limit — applies to the orchestrator graph and
        # all subagents that read Environment.get_recursion_limit().
        env["DEEP_AGENT_RECURSION_LIMIT"] = str(max(1, min(orch.recursion_limit, 10000)))

        # Per-run tool-call budget (soft nudge + hard graceful stop).
        env["TOOL_CALL_SOFT_BUDGET"] = str(max(0, orch.tool_call_soft_budget))
        env["TOOL_CALL_HARD_BUDGET"] = str(max(0, orch.tool_call_hard_budget))

        for srv in self.mcp_servers:
            if srv.id == "playwright-mcp" and srv.url:
                parsed = urlparse(srv.url)
                env["PLAYWRIGHT_MCP_HOST"] = parsed.hostname or "localhost"
                env["PLAYWRIGHT_MCP_PORT"] = str(parsed.port or 8931)

        ls = self.observability.langsmith
        env["LANGSMITH_TRACING"] = str(ls.enabled).lower()
        env["LANGSMITH_ENDPOINT"] = ls.endpoint
        env["LANGSMITH_API_KEY"] = ls.api_key
        env["LANGSMITH_PROJECT"] = ls.project

        env["LOG_LEVEL"] = self.observability.log_level

        x = self.exo
        env["EXO_MODE"] = x.mode
        env["EXO_REPO_URL"] = x.repo_url
        env["EXO_REF"] = x.repo_ref
        env["EXO_API_PORT"] = str(x.api_port)
        env["EXO_LIBP2P_PORT"] = str(x.libp2p_port)
        env["EXO_BASE_URL"] = x.effective_base_url
        env["EXO_MODEL_NAME"] = x.model_name
        env["EXO_NO_TERMINAL_WRAP"] = "1" if x.no_terminal_wrap else ""
        env["EXO_MAX_TOKENS"] = str(x.max_tokens)
        env["EXO_THINKING"] = "true" if x.enable_thinking else "false"

        ox = self.omlx
        env["OMLX_API_PORT"] = str(ox.api_port)
        env["OMLX_BASE_URL"] = ox.effective_base_url
        env["OMLX_MODEL_NAME"] = ox.model_name
        env["OMLX_CLI_PATH"] = ox.cli_path
        env["OMLX_THINKING"] = "true" if ox.thinking_enabled else "false"
        env["OMLX_MAX_TOKENS"] = str(ox.max_tokens)

        return env

    _AWS_CREDENTIAL_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")

    def apply_to_environ(self) -> None:
        """Write config values to ``os.environ``, overriding any ``.env`` or
        stale values.  Empty config values explicitly *remove* the env var
        so that the Settings UI is always authoritative."""
        for key, value in self.to_env_dict().items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

        self._apply_aws_credentials()
        # langsmith.utils.get_env_var is @lru_cache-decorated, so changes to
        # os.environ are invisible once it has been called.  Clear the cache
        # whenever we push new values so that tracing toggled in the Settings
        # UI (or at startup) is picked up on the very next LangChain call.
        try:
            from langsmith.utils import get_env_var as _ls_get_env_var
            _ls_get_env_var.cache_clear()
        except Exception:
            pass

    def _apply_aws_credentials(self) -> None:
        """Set or clear AWS credential env vars for Bedrock access-key auth.

        When not using Bedrock, all AWS credential env vars are cleared.
        """
        for key in self._AWS_CREDENTIAL_KEYS:
            os.environ.pop(key, None)

        a = self.llm.anthropic
        # Normalise legacy "anthropic_bedrock" spelling that older versions may have saved
        if a.model_provider == "anthropic_bedrock":
            a.model_provider = "bedrock"
        if self.llm.provider != "anthropic" or a.model_provider != "bedrock":
            return

        if a.aws_access_key_id and a.aws_secret_access_key:
            os.environ["AWS_ACCESS_KEY_ID"] = a.aws_access_key_id
            os.environ["AWS_SECRET_ACCESS_KEY"] = a.aws_secret_access_key
