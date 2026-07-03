"""Environment configuration module."""

import os
from dotenv import load_dotenv

import certifi
from utilities.logger import get_logger

logger = get_logger()


class Environment:
    """Environment configuration management."""

    COHERE_API_KEY = ""
    COHERE_MODEL = "command-a-03-2025"
    ENVIRONMENT_TYPE = "local"
    LOG_LEVEL = "INFO"
    MLX_MAX_TOKENS = "8192"
    MLX_TEMP = "0.0"
    MLX_REPETITION_PENALTY = "1.2"
    # How many recent tokens the repetition penalty considers.  mlx_lm defaults
    # to 20, which is too small to catch medium-period loops (e.g. a repeating
    # "|E0LU|E1LU|..." alternation); a wider window penalises them sooner.
    MLX_REPETITION_CONTEXT_SIZE = "60"
    MLX_VERBOSE = "false"
    MLX_THINKING = "false"
    MLX_PROMPT_CACHE = "false"
    MLX_SYSTEM_PROMPT_CACHE = "false"
    MLX_KV_BITS = ""
    MLX_KV_GROUP_SIZE = "64"
    MLX_NUM_DRAFT_TOKENS = "3"
    # Soft cap on the KV prefix cache, in tokens (0 = unbounded).  Primary
    # defence against unbounded memory growth in long autonomous sessions —
    # see ``MlxHfConfig.mlx_prompt_cache_max_tokens`` for the full rationale.
    MLX_PROMPT_CACHE_MAX_TOKENS = "32768"
    # Temperature the ToolLoopGuard requests for the recovery turn(s) after it
    # detects an identical-call loop.  Greedy decoding (temp 0) is a common
    # cause of such loops, so a one-shot bump lets the model break out.  0 =
    # disabled (keeps the configured temperature).  Only affects local MLX
    # models; a no-op for API providers.
    LOOP_RECOVERY_TEMPERATURE = "0.7"
    # Number of generations the recovery temperature bump applies to before it
    # auto-expires.
    LOOP_RECOVERY_TEMPERATURE_TURNS = "2"
    # Universal tool-loop-guard knobs (see tools.loop_guard).  ``WINDOW`` is the
    # recent-call ring-buffer size; ``MAX_NO_PROGRESS`` trips when that many
    # consecutive non-exempt calls return identical results; ``MAX_SUCCESS``
    # trips on that many identical successful calls; ``MAX_ESCALATIONS`` is the
    # total trip count after which the guard force-stops a runaway run (set 0
    # to disable the hard stop and only ever emit corrective messages).
    LOOP_GUARD_WINDOW = "8"
    LOOP_GUARD_MAX_NO_PROGRESS = "4"
    LOOP_GUARD_MAX_SUCCESS = "3"
    LOOP_GUARD_MAX_ESCALATIONS = "6"
    # Repeated-thought guard (see middleware.repeated_thought_guard).  Counts
    # consecutive AIMessages with an identical thought-text + tool-call
    # signature within a single agent invocation.  ``NUDGE_AT`` is how many
    # repeats trigger a temperature bump + corrective nudge; ``ABORT_AT`` is how
    # many repeats gracefully end the run (jump to end with a partial result so
    # a stuck subagent can't hang an orchestrator's parallel task batch).
    REPEAT_GUARD_NUDGE_AT = "3"
    REPEAT_GUARD_ABORT_AT = "10"
    # Longest repeating cycle the repeated-thought guard looks for.  ``1``
    # restricts detection to a consecutive-identical (period-1) streak; higher
    # values also catch short alternating loops (e.g. period-2 scroll/read).
    # ``0`` (or negative) falls back to period-1 only.
    REPEAT_GUARD_MAX_PERIOD = "4"
    # Turbo mode (oMLX-derived optimisations). See MlxHfConfig.turbo_* for
    # the full semantics.  Empty/"off" keeps the classic MLX path active.
    MLX_TURBO_LEVEL = "off"
    MLX_TURBO_SSD_DIR = ""
    MLX_TURBO_SSD_MAX_GB = "50"
    MLX_TURBO_TQ_BITS = "4"
    MLX_TURBO_BLOCK_SIZE = "256"
    # HuggingFace repo ID — MLX models are loaded from HuggingFace Hub (mlx-community)
    HF_LLM_MODEL_ID = "mlx-community/quantized-gemma-2b-it"
    HF_VLM_MODEL_ID = ""
    HF_DRAFT_LLM_MODEL_ID = ""
    # Optional: HF_TOKEN for gated/private models (huggingface_hub uses it automatically)
    # OpenAI (native API or Azure OpenAI)
    OPENAI_API_KEY = ""
    OPENAI_MODEL_NAME = "gpt-4o"
    OPENAI_MODEL_PROVIDER = "openai"  # "openai" | "azure"
    OPENAI_AZURE_ENDPOINT = ""
    OPENAI_AZURE_API_VERSION = "2024-12-01-preview"
    OPENAI_AZURE_DEPLOYMENT = ""
    OPENAI_MAX_TOKENS = "16384"
    OPENAI_TEMPERATURE = "0.0"
    # Anthropic Claude
    ANTHROPIC_API_KEY = ""
    ANTHROPIC_MODEL_NAME = "claude-sonnet-4-6"
    ANTHROPIC_MODEL_PROVIDER = "anthropic"
    ANTHROPIC_BEDROCK_REGION = "us-east-1"
    ANTHROPIC_BEDROCK_AUTH_MODE = "keys"
    ANTHROPIC_MAX_TOKENS = "8192"
    ANTHROPIC_THINKING_FLAG = "false"
    ANTHROPIC_THINKING_BUDGET = "2048"
    ANTHROPIC_TOOL_EFFICIENT_FLAG = "false"
    LLM_PROVIDER = "cohere"

    # Deep Agent orchestrator (optional separate provider / model)
    DEEP_AGENT_LLM_PROVIDER = ""
    DEEP_AGENT_MLX_MODEL_ID = ""
    DEEP_AGENT_MLX_MODEL_TYPE = "llm"  # "llm" or "vlm"
    # Maximum LangGraph steps per run — applies to the orchestrator AND all
    # subagents (general-purpose, web-voyager, computer-voyager).
    DEEP_AGENT_RECURSION_LIMIT = "10000"

    # Per-run tool-call budget.  ``recursion_limit`` (10 000) is too coarse to
    # catch a run that thrashes for hundreds of redundant browser/research
    # calls.  At the soft budget the agent gets a one-time nudge to converge; at
    # the hard budget the run ends gracefully with whatever it has.  0 disables
    # either threshold.
    TOOL_CALL_SOFT_BUDGET = "80"
    TOOL_CALL_HARD_BUDGET = "150"

    # exo distributed-inference cluster (OpenAI-compatible local API)
    EXO_BASE_URL = "http://127.0.0.1:52415"
    EXO_MODEL_NAME = ""
    EXO_THINKING = "false"
    EXO_MAX_TOKENS = "8192"

    # oMLX local inference server (OpenAI-compatible). External process,
    # installed by the user via Homebrew or .dmg — see
    # ``backend/omlx_provisioner.py`` for detection / install / lifecycle.
    OMLX_BASE_URL = "http://127.0.0.1:8000"
    OMLX_MODEL_NAME = ""
    OMLX_CLI_PATH = ""
    OMLX_API_PORT = "8000"
    OMLX_THINKING = "false"
    OMLX_MAX_TOKENS = "8192"

    # Repetition discouragement for OpenAI-compatible clients (exo / oMLX),
    # which — unlike the in-process MLX path — have no sampler-level
    # ``repetition_penalty``.  A small positive ``frequency_penalty`` makes a
    # degenerate "same sentence forever" loop far less likely at the API level;
    # the RepetitionGuard middleware remains the backstop.  0.0 disables.
    LLM_FREQUENCY_PENALTY = "0.3"
    LLM_PRESENCE_PENALTY = "0.0"

    # Prompt mode for the orchestrator.
    #   "auto"  → "lite" when the orchestrator runs on an OSS-local provider
    #             (mlx/exo), else "full".  This is the default and the value
    #             you almost always want.
    #   "full"  → force the long Claude-tuned prompt regardless of provider.
    #   "lite"  → force the short prompt regardless of provider (useful when
    #             benchmarking or when running a beefy local model that you
    #             trust to follow the long prompt).
    LOCAL_PROMPT_MODE = "auto"

    # Computer Voyager
    COMPUTER_VOYAGER_MAX_MESSAGES = "20"
    COMPUTER_VOYAGER_MAX_REPEAT = "3"
    COMPUTER_VOYAGER_MLX_MODEL_TYPE = "llm"  # "llm" or "vlm"

    # Playwright MCP client (connects to the Playwright MCP service)
    PLAYWRIGHT_MCP_HOST = "localhost"
    PLAYWRIGHT_MCP_PORT = "8931"
    PLAYWRIGHT_MAX_MESSAGES = "40"

    # macOS Accessibility (MacOSToolkit)
    SCAN_DEPTH = "30"
    SCAN_MAX_ELEMENTS = "500"
    SCAN_MAX_WORKERS = "3"
    AX_IPC_TIMEOUT = "5"
    SCAN_TIME_BUDGET = "20"
    AX_BRIDGE_INIT_DELAY = "3"

    @classmethod
    def load(cls):
        """Load environment variables from .env file and set defaults."""
        load_dotenv()

        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                default_value = getattr(cls, attr)
                env_value = os.getenv(attr, default_value)
                setattr(cls, attr, env_value)
                os.environ[attr] = str(env_value)

    @classmethod
    def get_cohere_api_key(cls) -> str:
        """Get Cohere API key."""
        return os.getenv("COHERE_API_KEY", cls.COHERE_API_KEY)

    @classmethod
    def get_cohere_model(cls) -> str:
        """Get Cohere model name."""
        return os.getenv("COHERE_MODEL", cls.COHERE_MODEL)

    @classmethod
    def get_environment_type(cls) -> str:
        """Get environment type."""
        return os.getenv("ENVIRONMENT_TYPE", cls.ENVIRONMENT_TYPE)

    @classmethod
    def is_local(cls) -> bool:
        """Check if running in local environment."""
        return cls.get_environment_type() in ("", "local")

    @classmethod
    def get_mlx_max_tokens(cls) -> int:
        """Get MLX max tokens for generation."""
        return int(os.getenv("MLX_MAX_TOKENS", cls.MLX_MAX_TOKENS))

    @classmethod
    def get_mlx_temp(cls) -> float:
        """Get MLX temperature for generation (0.0 = deterministic)."""
        return float(os.getenv("MLX_TEMP", cls.MLX_TEMP))

    @classmethod
    def get_mlx_verbose(cls) -> bool:
        """Get MLX agent verbose mode (prints Thought/Action/Observation)."""
        val = os.getenv("MLX_VERBOSE", cls.MLX_VERBOSE)
        return str(val).lower() in ("true", "1", "yes")

    @classmethod
    def get_mlx_thinking(cls) -> bool:
        """Get whether chain-of-thought thinking is enabled for MLX models.

        Only affects models that support a thinking toggle (e.g. Qwen3).
        DeepSeek-R1 always thinks regardless of this setting.
        """
        val = os.getenv("MLX_THINKING", cls.MLX_THINKING)
        return str(val).lower() in ("true", "1", "yes")

    @classmethod
    def get_mlx_prompt_cache(cls) -> bool:
        """Get whether KV prompt caching is enabled for MLX text models.

        When ``True``, each ``ChatMLXText`` instance maintains a running KV
        cache across calls, eliminating repeated prefill of the system prompt
        and prior turns.  Disabled by default; enable for long-running agent
        sessions where the same model instance handles many turns.
        """
        val = os.getenv("MLX_PROMPT_CACHE", cls.MLX_PROMPT_CACHE)
        return str(val).lower() in ("true", "1", "yes")

    @classmethod
    def get_mlx_system_prompt_cache(cls) -> bool:
        """Get whether prefix-reuse caching is enabled for the static system/tool prompt.

        When ``True`` (and ``MLX_PROMPT_CACHE`` is also ``True``), each turn
        compares the tokenized prompt with the previous turn's tokens, trims
        the KV cache back to the longest common prefix, and only feeds the new
        suffix tokens to the model — avoiding repeated prefill of the system
        prompt and tool definitions.
        """
        val = os.getenv("MLX_SYSTEM_PROMPT_CACHE", cls.MLX_SYSTEM_PROMPT_CACHE)
        return str(val).lower() in ("true", "1", "yes")

    @classmethod
    def get_mlx_kv_bits(cls) -> int | None:
        """Get KV cache quantization bits (4 or 8). Empty/unset = full precision."""
        val = os.getenv("MLX_KV_BITS", cls.MLX_KV_BITS).strip()
        if not val:
            return None
        return int(val)

    @classmethod
    def get_mlx_kv_group_size(cls) -> int:
        """Get KV cache quantization group size (default 64).

        Smaller values (e.g. 32) preserve more quality at the cost of slightly
        more memory for the scale factors.  Only used when ``MLX_KV_BITS`` is set.
        """
        return int(os.getenv("MLX_KV_GROUP_SIZE", cls.MLX_KV_GROUP_SIZE))

    @classmethod
    def get_mlx_num_draft_tokens(cls) -> int:
        """Get number of tokens the draft model proposes per speculative step (default 3).

        Increasing this can raise throughput when acceptance rates are high,
        but reduces gains when the draft and target models disagree frequently.
        Typically 3-6 is optimal; tune against your model pair and prompt mix.
        """
        return int(os.getenv("MLX_NUM_DRAFT_TOKENS", cls.MLX_NUM_DRAFT_TOKENS))

    @classmethod
    def get_mlx_prompt_cache_max_tokens(cls) -> int:
        """Soft cap on the KV prefix cache size, measured in tokens.

        After each generation, ``ChatMLXText`` checks the cumulative cache
        offset; when it exceeds this value the cache is trimmed (or fully
        rebuilt for non-trimmable layer types) so long autonomous sessions
        don't OOM the host.  ``0`` disables the cap (legacy unbounded
        behaviour).  Default ``32768`` ≈ 1 GB on a 7B 4-bit model.
        """
        try:
            return max(0, int(os.getenv("MLX_PROMPT_CACHE_MAX_TOKENS", cls.MLX_PROMPT_CACHE_MAX_TOKENS)))
        except ValueError:
            return 32768

    # ── Turbo mode (oMLX-derived optimisations) ──────────────────────────

    # Only levels with a real implementation map through; anything else
    # collapses to ``off`` so a stale config never forces an unsupported
    # code path.  Extend this tuple as new levels land.
    _TURBO_LEVELS = ("off", "basic", "cache", "ssd")

    @classmethod
    def get_mlx_turbo_level(cls) -> str:
        """Return the configured MLX turbo level (``off`` disables turbo).

        Unknown values fall back to ``off`` so a stale config never forces a
        user onto an unsupported code path.
        """
        val = os.getenv("MLX_TURBO_LEVEL", cls.MLX_TURBO_LEVEL).strip().lower()
        return val if val in cls._TURBO_LEVELS else "off"

    @classmethod
    def get_mlx_turbo_ssd_dir(cls) -> str:
        """Directory for the SSD cold-tier KV cache (empty ⇒ auto)."""
        return os.getenv("MLX_TURBO_SSD_DIR", cls.MLX_TURBO_SSD_DIR)

    @classmethod
    def get_mlx_turbo_ssd_max_gb(cls) -> int:
        """Soft cap (GB) on the SSD cache footprint."""
        try:
            return int(os.getenv("MLX_TURBO_SSD_MAX_GB", cls.MLX_TURBO_SSD_MAX_GB))
        except ValueError:
            return 50

    @classmethod
    def get_mlx_turbo_tq_bits(cls) -> int:
        """TurboQuant KV bits (4 or 8). Only used at turbo_level=max."""
        try:
            bits = int(os.getenv("MLX_TURBO_TQ_BITS", cls.MLX_TURBO_TQ_BITS))
        except ValueError:
            return 4
        return bits if bits in (4, 8) else 4

    @classmethod
    def get_mlx_turbo_block_size(cls) -> int:
        """Paged-cache block size (tokens per block)."""
        try:
            return int(os.getenv("MLX_TURBO_BLOCK_SIZE", cls.MLX_TURBO_BLOCK_SIZE))
        except ValueError:
            return 256

    @classmethod
    def get_mlx_repetition_penalty(cls) -> float:
        """Get MLX repetition penalty to reduce repeated tokens (1.0 = off)."""
        return float(os.getenv("MLX_REPETITION_PENALTY", cls.MLX_REPETITION_PENALTY))

    @classmethod
    def get_mlx_repetition_context_size(cls) -> int:
        """Get the window (in tokens) the repetition penalty looks back over."""
        try:
            return int(os.getenv("MLX_REPETITION_CONTEXT_SIZE", cls.MLX_REPETITION_CONTEXT_SIZE))
        except ValueError:
            return 60

    @classmethod
    def get_hf_llm_model_id(cls) -> str:
        """Get HuggingFace repo ID for MLX model (models load from HuggingFace Hub)."""
        return os.getenv("HF_LLM_MODEL_ID", cls.HF_LLM_MODEL_ID)

    @classmethod
    def get_hf_vlm_model_id(cls) -> str | None:
        """Get HuggingFace repo ID for MLX vision-language model.

        Returns ``None`` when ``HF_VLM_MODEL_ID`` is unset or empty,
        signalling that no VLM should be loaded.
        """
        val = os.getenv("HF_VLM_MODEL_ID", cls.HF_VLM_MODEL_ID).strip()
        return val or None

    @classmethod
    def get_hf_draft_llm_model_id(cls) -> str | None:
        """Get optional HuggingFace repo ID for a draft model (speculative decoding).

        When set, ``ChatMLXText`` loads a small draft model alongside the main
        model to accelerate generation via speculative decoding.
        Returns ``None`` when unset or empty.
        """
        val = os.getenv("HF_DRAFT_LLM_MODEL_ID", cls.HF_DRAFT_LLM_MODEL_ID).strip()
        return val or None

    @classmethod
    def get_anthropic_api_key(cls) -> str:
        """Get Anthropic API key for Claude."""
        return os.getenv("ANTHROPIC_API_KEY", cls.ANTHROPIC_API_KEY)

    @classmethod
    def get_anthropic_model_name(cls) -> str:
        """Get Anthropic Claude model name (e.g. claude-sonnet-4-6)."""
        return os.getenv("ANTHROPIC_MODEL_NAME", cls.ANTHROPIC_MODEL_NAME)

    @classmethod
    def get_anthropic_model_provider(cls) -> str:
        """Get Anthropic provider: anthropic, bedrock, vertex."""
        return os.getenv("ANTHROPIC_MODEL_PROVIDER", cls.ANTHROPIC_MODEL_PROVIDER)

    @classmethod
    def get_exo_base_url(cls) -> str:
        """Base URL of the exo cluster's OpenAI-compatible API
        (e.g. ``http://127.0.0.1:52415``).  ``/v1`` is appended by the
        client; provide the bare host:port here."""
        return (os.getenv("EXO_BASE_URL", cls.EXO_BASE_URL) or "").rstrip("/")

    @classmethod
    def get_exo_model_name(cls) -> str:
        """Default exo model id (e.g. ``mlx-community/Qwen3.5-9B-4bit``).
        Picked in Settings → LLM → Models when provider is ``exo``."""
        return os.getenv("EXO_MODEL_NAME", cls.EXO_MODEL_NAME)

    @classmethod
    def get_exo_thinking(cls) -> bool:
        """Whether chain-of-thought thinking is enabled for exo models.

        Sent per request as ``enable_thinking`` (exo maps it internally to
        ``reasoning_effort``) by ``model_factory.create_llm``.  Only affects
        models whose chat template supports a thinking switch (Qwen3,
        DeepSeek V3.1, GLM-4.x).  Defaults off for lowest latency."""
        val = os.getenv("EXO_THINKING", cls.EXO_THINKING)
        return str(val).lower() in ("true", "1", "yes")

    @classmethod
    def get_exo_max_tokens(cls) -> int:
        """Maximum tokens generated per exo response.

        Sent per request as ``max_tokens`` by ``model_factory.create_llm``.
        Bounds runaway generations on models that fail to emit a stop token
        (exo otherwise falls back to the model's full context window).  Falls
        back to the class default (8192) if unset or unparseable."""
        val = os.getenv("EXO_MAX_TOKENS", cls.EXO_MAX_TOKENS)
        try:
            n = int(val)
            return n if n > 0 else int(cls.EXO_MAX_TOKENS)
        except (TypeError, ValueError):
            return int(cls.EXO_MAX_TOKENS)

    @classmethod
    def get_omlx_base_url(cls) -> str:
        """Base URL of the local oMLX server's OpenAI-compatible API
        (e.g. ``http://127.0.0.1:8000``).  ``/v1`` is appended by the
        client; provide the bare host:port here."""
        return (os.getenv("OMLX_BASE_URL", cls.OMLX_BASE_URL) or "").rstrip("/")

    @classmethod
    def get_omlx_model_name(cls) -> str:
        """Default oMLX model id served by the local server. Picked in
        Settings → LLM → Models when provider is ``omlx``."""
        return os.getenv("OMLX_MODEL_NAME", cls.OMLX_MODEL_NAME)

    @classmethod
    def get_omlx_cli_path(cls) -> str:
        """Optional explicit path to the ``omlx`` CLI. Empty means resolve
        via the system PATH (handles Homebrew default install)."""
        return os.getenv("OMLX_CLI_PATH", cls.OMLX_CLI_PATH)

    @classmethod
    def get_omlx_thinking(cls) -> bool:
        """Whether chain-of-thought thinking is enabled for oMLX models.

        oMLX exposes no global admin toggle for thinking, so this is sent
        per request as ``chat_template_kwargs={"enable_thinking": <bool>}``
        by ``model_factory.create_llm``.  Only affects models whose chat
        template supports a thinking switch (e.g. Qwen3)."""
        val = os.getenv("OMLX_THINKING", cls.OMLX_THINKING)
        return str(val).lower() in ("true", "1", "yes")

    @classmethod
    def get_omlx_max_tokens(cls) -> int:
        """Maximum tokens generated per oMLX response.

        Sent per request as ``max_tokens`` by ``model_factory.create_llm``.
        Bounds runaway generations on models that fail to emit a stop
        token (the server default ``sampling.max_tokens`` is 32768, which
        can produce ~10-minute completions).  Falls back to the class
        default (8192) if unset or unparseable."""
        val = os.getenv("OMLX_MAX_TOKENS", cls.OMLX_MAX_TOKENS)
        try:
            n = int(val)
            return n if n > 0 else int(cls.OMLX_MAX_TOKENS)
        except (TypeError, ValueError):
            return int(cls.OMLX_MAX_TOKENS)

    @classmethod
    def get_llm_frequency_penalty(cls) -> float:
        """``frequency_penalty`` sent to OpenAI-compatible clients (exo/oMLX).

        A small positive value discourages verbatim repetition at the API
        level (these clients have no in-process ``repetition_penalty``).
        Falls back to the class default if unset or unparseable; clamped to
        the OpenAI-supported ``[-2.0, 2.0]`` range."""
        val = os.getenv("LLM_FREQUENCY_PENALTY", cls.LLM_FREQUENCY_PENALTY)
        try:
            return max(-2.0, min(2.0, float(val)))
        except (TypeError, ValueError):
            return float(cls.LLM_FREQUENCY_PENALTY)

    @classmethod
    def get_llm_presence_penalty(cls) -> float:
        """``presence_penalty`` sent to OpenAI-compatible clients (exo/oMLX).

        Falls back to the class default if unset or unparseable; clamped to
        the OpenAI-supported ``[-2.0, 2.0]`` range."""
        val = os.getenv("LLM_PRESENCE_PENALTY", cls.LLM_PRESENCE_PENALTY)
        try:
            return max(-2.0, min(2.0, float(val)))
        except (TypeError, ValueError):
            return float(cls.LLM_PRESENCE_PENALTY)

    @classmethod
    def get_anthropic_bedrock_region(cls) -> str:
        """Get AWS region for Anthropic Bedrock."""
        return os.getenv("ANTHROPIC_BEDROCK_REGION", cls.ANTHROPIC_BEDROCK_REGION)

    @classmethod
    def get_anthropic_bedrock_auth_mode(cls) -> str:
        """Bedrock credential mode: ``keys`` (env access keys) or ``sso`` (default chain)."""
        return os.getenv("ANTHROPIC_BEDROCK_AUTH_MODE", cls.ANTHROPIC_BEDROCK_AUTH_MODE).lower().strip()

    @classmethod
    def get_anthropic_max_tokens(cls) -> int:
        """Get max tokens for Anthropic model responses."""
        return int(os.getenv("ANTHROPIC_MAX_TOKENS", cls.ANTHROPIC_MAX_TOKENS))

    @classmethod
    def get_anthropic_thinking_flag(cls) -> bool:
        """Get whether extended thinking is enabled for Anthropic models."""
        val = os.getenv("ANTHROPIC_THINKING_FLAG", cls.ANTHROPIC_THINKING_FLAG)
        return str(val).lower() in ("true", "1", "yes")

    @classmethod
    def get_anthropic_thinking_budget(cls) -> int:
        """Get token budget for Anthropic extended thinking."""
        return int(os.getenv("ANTHROPIC_THINKING_BUDGET", cls.ANTHROPIC_THINKING_BUDGET))

    @classmethod
    def get_anthropic_tool_efficient_flag(cls) -> bool:
        """Get whether tool-use efficiency mode is enabled."""
        val = os.getenv("ANTHROPIC_TOOL_EFFICIENT_FLAG", cls.ANTHROPIC_TOOL_EFFICIENT_FLAG)
        return str(val).lower() in ("true", "1", "yes")

    @classmethod
    def get_openai_api_key(cls) -> str:
        """Get OpenAI API key (used for both native and Azure OpenAI)."""
        return os.getenv("OPENAI_API_KEY", cls.OPENAI_API_KEY)

    @classmethod
    def get_openai_model_name(cls) -> str:
        """Get OpenAI model name (e.g. gpt-4o, o1-preview, o3-mini)."""
        return os.getenv("OPENAI_MODEL_NAME", cls.OPENAI_MODEL_NAME)

    @classmethod
    def get_openai_model_provider(cls) -> str:
        """Get OpenAI provider mode: 'openai' (native API) or 'azure' (Azure OpenAI)."""
        return os.getenv("OPENAI_MODEL_PROVIDER", cls.OPENAI_MODEL_PROVIDER).lower().strip()

    @classmethod
    def get_openai_azure_endpoint(cls) -> str:
        """Get Azure OpenAI endpoint URL (e.g. https://<resource>.openai.azure.com)."""
        return os.getenv("OPENAI_AZURE_ENDPOINT", cls.OPENAI_AZURE_ENDPOINT)

    @classmethod
    def get_openai_azure_api_version(cls) -> str:
        """Get Azure OpenAI API version string (e.g. 2024-12-01-preview)."""
        return os.getenv("OPENAI_AZURE_API_VERSION", cls.OPENAI_AZURE_API_VERSION)

    @classmethod
    def get_openai_azure_deployment(cls) -> str:
        """Get Azure OpenAI deployment name. Falls back to model name when empty."""
        val = os.getenv("OPENAI_AZURE_DEPLOYMENT", cls.OPENAI_AZURE_DEPLOYMENT).strip()
        return val or cls.get_openai_model_name()

    @classmethod
    def get_openai_max_tokens(cls) -> int:
        """Get max tokens for OpenAI model responses."""
        return int(os.getenv("OPENAI_MAX_TOKENS", cls.OPENAI_MAX_TOKENS))

    @classmethod
    def get_openai_temperature(cls) -> float:
        """Get temperature for OpenAI model (0.0 = deterministic)."""
        return float(os.getenv("OPENAI_TEMPERATURE", cls.OPENAI_TEMPERATURE))

    @classmethod
    def get_llm_provider(cls) -> str:
        """Get chat API provider: cohere or anthropic."""
        return os.getenv("LLM_PROVIDER", cls.LLM_PROVIDER).lower().strip()

    # ── Open-source local provider detection ─────────────────────────────
    #
    # ``mlx`` and ``exo`` are the two providers that run weights on the
    # user's machine.  These models typically have small context windows
    # and are sensitive to long instruction prompts, so the orchestrator
    # uses a slimmer prompt when it talks to them.  The helpers below are
    # the single source of truth for that decision so we don't sprinkle
    # ``provider in ("mlx", "exo")`` checks across the code base.

    _OSS_LOCAL_PROVIDERS = frozenset({"mlx", "exo", "omlx"})

    @classmethod
    def is_oss_local_provider(cls, provider: str | None = None) -> bool:
        """Return True when *provider* is an open-source local LLM family.

        Pass ``None`` (the default) to evaluate the main ``LLM_PROVIDER``.
        """
        prov = (provider if provider is not None else cls.get_llm_provider()) or ""
        return prov.lower().strip() in cls._OSS_LOCAL_PROVIDERS

    @classmethod
    def is_orchestrator_oss_local(cls) -> bool:
        """Return True when the *orchestrator* is on an OSS-local provider.

        Honours ``DEEP_AGENT_LLM_PROVIDER`` (split orchestrator/subagent
        providers) before falling back to ``LLM_PROVIDER``.
        """
        da = cls.get_deep_agent_llm_provider()
        return cls.is_oss_local_provider(da if da else cls.get_llm_provider())

    @classmethod
    def get_local_prompt_mode(cls) -> str:
        """Return ``"auto" | "full" | "lite"`` (lower-case, validated).

        Unknown values fall back to ``"auto"``.
        """
        val = os.getenv("LOCAL_PROMPT_MODE", cls.LOCAL_PROMPT_MODE).strip().lower()
        if val not in ("auto", "full", "lite"):
            return "auto"
        return val

    @classmethod
    def use_lite_orchestrator_prompt(cls) -> bool:
        """Resolve ``LOCAL_PROMPT_MODE`` + provider to a lite-or-full bool.

        ``auto`` → full (all providers, including mlx/omlx/exo).
        ``full`` / ``lite`` → forced regardless of provider.
        """
        mode = cls.get_local_prompt_mode()
        if mode == "lite":
            return True
        return False

    @classmethod
    def get_deep_agent_llm_provider(cls) -> str | None:
        """Get optional LLM provider for the DeepAgent orchestrator.

        When set (e.g. ``"anthropic"``), the orchestrator uses a different
        provider than the subagents.  Empty/unset = use ``LLM_PROVIDER``.
        """
        val = os.getenv("DEEP_AGENT_LLM_PROVIDER", cls.DEEP_AGENT_LLM_PROVIDER).strip()
        return val.lower() if val else None

    @classmethod
    def get_deep_agent_mlx_model_id(cls) -> str | None:
        """Get optional MLX model ID for the DeepAgent orchestrator.

        When set, the orchestrator loads its own MLX model instead of sharing
        ``HF_LLM_MODEL_ID`` with subagents.  Empty/unset = use the default LLM.
        """
        val = os.getenv("DEEP_AGENT_MLX_MODEL_ID", cls.DEEP_AGENT_MLX_MODEL_ID).strip()
        return val or None

    @classmethod
    def get_deep_agent_mlx_model_type(cls) -> str:
        """Get MLX model type for the DeepAgent orchestrator.

        Returns ``"llm"`` (text-only) or ``"vlm"`` (vision-language).
        Text-only models cannot interpret images, so image-returning tools
        (e.g. ``browser_take_screenshot``) are excluded when set to ``"llm"``.
        """
        return os.getenv("DEEP_AGENT_MLX_MODEL_TYPE", cls.DEEP_AGENT_MLX_MODEL_TYPE).strip().lower()

    @classmethod
    def get_recursion_limit(cls) -> int:
        """Maximum LangGraph steps per run (1–10 000).

        Applied to the orchestrator and all subagents (general-purpose,
        web-voyager, computer-voyager).  Configured via
        Settings → Advanced → Agent execution or the
        ``DEEP_AGENT_RECURSION_LIMIT`` environment variable.
        """
        try:
            val = int(os.getenv("DEEP_AGENT_RECURSION_LIMIT", cls.DEEP_AGENT_RECURSION_LIMIT))
            return max(1, min(val, 10000))
        except (ValueError, TypeError):
            return 1000

    @classmethod
    def get_tool_call_soft_budget(cls) -> int:
        """Tool calls per run after which the agent is nudged to converge.

        ``0`` disables the soft nudge.  Configured via the
        ``TOOL_CALL_SOFT_BUDGET`` environment variable."""
        try:
            return max(0, int(os.getenv("TOOL_CALL_SOFT_BUDGET", cls.TOOL_CALL_SOFT_BUDGET)))
        except (ValueError, TypeError):
            return int(cls.TOOL_CALL_SOFT_BUDGET)

    @classmethod
    def get_tool_call_hard_budget(cls) -> int:
        """Tool calls per run after which the run ends gracefully.

        ``0`` disables the hard stop.  Configured via the
        ``TOOL_CALL_HARD_BUDGET`` environment variable."""
        try:
            return max(0, int(os.getenv("TOOL_CALL_HARD_BUDGET", cls.TOOL_CALL_HARD_BUDGET)))
        except (ValueError, TypeError):
            return int(cls.TOOL_CALL_HARD_BUDGET)

    @classmethod
    def get_computer_voyager_max_messages(cls) -> int:
        """Get max messages kept in the LLM context for the computer voyager agent."""
        return int(os.getenv("COMPUTER_VOYAGER_MAX_MESSAGES", cls.COMPUTER_VOYAGER_MAX_MESSAGES))

    @classmethod
    def get_computer_voyager_max_repeat(cls) -> int:
        """Get max consecutive identical tool calls before force-stopping the agent."""
        return int(os.getenv("COMPUTER_VOYAGER_MAX_REPEAT", cls.COMPUTER_VOYAGER_MAX_REPEAT))

    @classmethod
    def get_computer_voyager_mlx_model_type(cls) -> str:
        """Return ``'vlm'`` or ``'llm'`` for the computer voyager MLX backend."""
        val = os.getenv("COMPUTER_VOYAGER_MLX_MODEL_TYPE", cls.COMPUTER_VOYAGER_MLX_MODEL_TYPE)
        return val.strip().lower()

    # ── Playwright MCP ──────────────────────────────────────────────────

    @classmethod
    def get_playwright_mcp_host(cls) -> str:
        return os.getenv("PLAYWRIGHT_MCP_HOST", cls.PLAYWRIGHT_MCP_HOST)

    @classmethod
    def get_playwright_mcp_port(cls) -> int:
        return int(os.getenv("PLAYWRIGHT_MCP_PORT", cls.PLAYWRIGHT_MCP_PORT))

    @classmethod
    def get_playwright_max_messages(cls) -> int:
        """Get max messages kept in the LLM context for Playwright MCP pruning."""
        return int(os.getenv("PLAYWRIGHT_MAX_MESSAGES", cls.PLAYWRIGHT_MAX_MESSAGES))

    @classmethod
    def get_loop_recovery_temperature(cls) -> float:
        """Temperature the loop guard requests after detecting a call loop.

        ``0`` disables the bump.  Only the local MLX path honours it (see
        ``chat_models.mlx._shared.request_temperature_bump``); API providers
        ignore it.
        """
        return float(
            os.getenv("LOOP_RECOVERY_TEMPERATURE", cls.LOOP_RECOVERY_TEMPERATURE)
        )

    @classmethod
    def get_loop_recovery_temperature_turns(cls) -> int:
        """Number of generations the recovery temperature bump applies to."""
        return int(
            os.getenv(
                "LOOP_RECOVERY_TEMPERATURE_TURNS",
                cls.LOOP_RECOVERY_TEMPERATURE_TURNS,
            )
        )

    @classmethod
    def get_loop_guard_window(cls) -> int:
        """Size of the loop guard's recent-call ring buffer."""
        return max(2, int(os.getenv("LOOP_GUARD_WINDOW", cls.LOOP_GUARD_WINDOW)))

    @classmethod
    def get_loop_guard_max_no_progress(cls) -> int | None:
        """Consecutive identical-result calls that trip the no-progress guard.

        ``0`` (or negative) disables no-progress detection."""
        val = int(
            os.getenv("LOOP_GUARD_MAX_NO_PROGRESS", cls.LOOP_GUARD_MAX_NO_PROGRESS)
        )
        return val if val > 0 else None

    @classmethod
    def get_loop_guard_max_success(cls) -> int | None:
        """Identical successful calls that trip the success-loop guard.

        ``0`` (or negative) disables success-loop detection."""
        val = int(
            os.getenv("LOOP_GUARD_MAX_SUCCESS", cls.LOOP_GUARD_MAX_SUCCESS)
        )
        return val if val > 0 else None

    @classmethod
    def get_loop_guard_max_escalations(cls) -> int | None:
        """Total trips after which the guard force-stops a runaway run.

        ``0`` (or negative) disables the cooperative hard stop; the guard then
        only ever emits corrective messages."""
        val = int(
            os.getenv("LOOP_GUARD_MAX_ESCALATIONS", cls.LOOP_GUARD_MAX_ESCALATIONS)
        )
        return val if val > 0 else None

    @classmethod
    def get_repeat_guard_nudge_at(cls) -> int | None:
        """Consecutive identical thoughts that trigger a nudge + temp bump.

        ``0`` (or negative) disables the nudge stage; the guard then only ever
        applies the hard abort at :meth:`get_repeat_guard_abort_at`."""
        val = int(
            os.getenv("REPEAT_GUARD_NUDGE_AT", cls.REPEAT_GUARD_NUDGE_AT)
        )
        return val if val > 0 else None

    @classmethod
    def get_repeat_guard_abort_at(cls) -> int | None:
        """Consecutive identical thoughts after which the run ends gracefully.

        ``0`` (or negative) disables the cooperative abort; the guard then only
        ever nudges (relying on ``recursion_limit`` as the hard stop)."""
        val = int(
            os.getenv("REPEAT_GUARD_ABORT_AT", cls.REPEAT_GUARD_ABORT_AT)
        )
        return val if val > 0 else None

    @classmethod
    def get_repeat_guard_max_period(cls) -> int:
        """Longest repeating AIMessage cycle the guard scans for.

        ``1`` keeps detection to a consecutive-identical streak; higher values
        also catch short alternating loops (e.g. a period-2 scroll/read cycle).
        ``0`` or negative is clamped to ``1`` (period-1 only)."""
        val = int(
            os.getenv("REPEAT_GUARD_MAX_PERIOD", cls.REPEAT_GUARD_MAX_PERIOD)
        )
        return val if val > 0 else 1

    @classmethod
    def get_scan_depth(cls) -> int:
        """Get max AX tree walk depth for macOS accessibility scanning."""
        return int(os.getenv("SCAN_DEPTH", cls.SCAN_DEPTH))

    @classmethod
    def get_scan_max_elements(cls) -> int:
        """Get max UI elements returned per AX scan."""
        return int(os.getenv("SCAN_MAX_ELEMENTS", cls.SCAN_MAX_ELEMENTS))

    @classmethod
    def get_scan_max_workers(cls) -> int:
        """Get max thread-pool workers for parallel AX tree walking."""
        return int(os.getenv("SCAN_MAX_WORKERS", cls.SCAN_MAX_WORKERS))

    @classmethod
    def get_ax_ipc_timeout(cls) -> float:
        """Get timeout (seconds) for individual AX IPC calls."""
        return float(os.getenv("AX_IPC_TIMEOUT", cls.AX_IPC_TIMEOUT))

    @classmethod
    def get_scan_time_budget(cls) -> float:
        """Get wall-clock budget (seconds) for a single AX tree scan/walk.

        Bounds how long get_screen_controls can spend traversing a large
        accessibility tree (e.g. Electron apps like Slack) before returning
        whatever controls it has gathered so far.
        """
        return float(os.getenv("SCAN_TIME_BUDGET", cls.SCAN_TIME_BUDGET))

    @classmethod
    def get_ax_bridge_init_delay(cls) -> float:
        """Get sleep duration (seconds) after activating the Chromium AX bridge.

        This one-time delay (per PID) gives Electron/Chromium apps time to
        populate their accessibility tree after AXManualAccessibility is set.
        3s is sufficient for most apps; increase if controls are missing on
        first scan.
        """
        return float(os.getenv("AX_BRIDGE_INIT_DELAY", cls.AX_BRIDGE_INIT_DELAY))

    @classmethod
    def validate(cls) -> bool:
        """Validate required environment variables for Cohere API."""
        api_key = cls.get_cohere_api_key()
        if not api_key:
            logger.error("COHERE_API_KEY is not set")
            return False
        return True

    @classmethod
    def validate_anthropic(cls) -> bool:
        """Validate required environment variables for Anthropic API."""
        api_key = cls.get_anthropic_api_key()
        if not api_key:
            logger.error("ANTHROPIC_API_KEY is not set")
            return False
        return True

    @classmethod
    def validate_openai(cls) -> bool:
        """Validate required environment variables for OpenAI API."""
        if cls.get_openai_model_provider() == "azure":
            if not cls.get_openai_azure_endpoint():
                logger.error("OPENAI_AZURE_ENDPOINT is not set for Azure OpenAI")
                return False
        else:
            if not cls.get_openai_api_key():
                logger.error("OPENAI_API_KEY is not set")
                return False
        return True

    @classmethod
    def validate_chat(cls) -> bool:
        """Validate required env vars for the active chat provider (cohere|anthropic|openai)."""
        provider = cls.get_llm_provider()
        if provider == "anthropic":
            return cls.validate_anthropic()
        if provider == "openai":
            return cls.validate_openai()
        return cls.validate()
