"""Settings-related API routes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from pathlib import Path

from fastapi import APIRouter

from backend.config import (
    AppConfig,
    _SECRET_FIELDS,
    _dict_get_path,
    _dict_set_path,
    _model_get_path,
)
from backend.mcp_manager import reset_circuit_breaker
from backend.schemas import TestConnectionRequest, TestConnectionResponse
from backend.state import mcp_mgr, session_mgr

logger = logging.getLogger(__name__)

# Models that only accept temperature=1 (e.g. GPT-5, o-series reasoning models).
_FIXED_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o2", "o3", "o4")


def _openai_temperature(model_name: str, default: float = 0.0) -> float:
    """Return a safe temperature for an OpenAI/Azure model.

    GPT-5 and o-series reasoning models reject any temperature != 1.
    For those models we coerce to 1 regardless of the configured value.
    """
    name = (model_name or "").lower().strip()
    for prefix in _FIXED_TEMPERATURE_PREFIXES:
        if name == prefix or name.startswith(prefix + "-") or name.startswith(prefix + "."):
            return 1.0
    return default

router = APIRouter(prefix="/api", tags=["settings"])


@router.get("/health")
async def health():
    return {"status": "ok"}


_REDACTED_PLACEHOLDER = "••••"


@router.get("/settings")
async def get_settings():
    cfg = await AppConfig.aload()
    data = cfg.model_dump()
    # Never return secret values to the frontend — mask each known
    # secret field with a placeholder when set, or "" when unset.
    for path, _account in _SECRET_FIELDS:
        value = _dict_get_path(data, path)
        _dict_set_path(data, path, _REDACTED_PLACEHOLDER if value else "")
    return data


_CLAUDE_HOOK_SERVER_ID = "claude-eval-hook"
_OPENCLAW_SERVER_ID = "openclaw-eval-hook"


@router.put("/settings")
async def update_settings(payload: dict):
    existing = await AppConfig.aload()
    # A placeholder coming back means "unchanged" — restore the real value
    # from the (vault-hydrated) existing config before persisting.
    for path, _account in _SECRET_FIELDS:
        if _dict_get_path(payload, path) == _REDACTED_PLACEHOLDER:
            _dict_set_path(payload, path, _model_get_path(existing, path))
    cfg = AppConfig.model_validate(payload)

    claude_hook_changed = existing.claude_hook.enabled != cfg.claude_hook.enabled
    if claude_hook_changed:
        _sync_hook_mcp_server(cfg, _CLAUDE_HOOK_SERVER_ID, cfg.claude_hook.enabled)

    openclaw_changed = existing.openclaw.enabled != cfg.openclaw.enabled
    if openclaw_changed:
        _sync_hook_mcp_server(cfg, _OPENCLAW_SERVER_ID, cfg.openclaw.enabled)

    oc_watcher_changed = (
        existing.openclaw.watcher_enabled != cfg.openclaw.watcher_enabled
        or existing.openclaw.enabled != cfg.openclaw.enabled
    )

    await cfg.asave()
    cfg.apply_to_environ()

    llm_changed = existing.llm.model_dump() != cfg.llm.model_dump()
    exo_model_changed = existing.exo.model_name != cfg.exo.model_name
    if llm_changed or exo_model_changed:
        logger.info("LLM/EXO model config changed — rebuilding active sessions")
        await session_mgr.refresh_tools(cfg)

    if claude_hook_changed:
        await _reconnect_hook_server(cfg, _CLAUDE_HOOK_SERVER_ID)
    if openclaw_changed:
        await _reconnect_hook_server(cfg, _OPENCLAW_SERVER_ID)

    if oc_watcher_changed:
        await _sync_openclaw_watcher(cfg)

    return {"status": "saved"}


def _sync_hook_mcp_server(cfg: AppConfig, server_id: str, enabled: bool) -> None:
    """Keep an eval-hook MCPServerConfig in sync with its integration toggle."""
    for srv in cfg.mcp_servers:
        if srv.id == server_id:
            srv.enabled = enabled
            return


async def _reconnect_hook_server(cfg: AppConfig, server_id: str) -> None:
    """Start or stop an eval-hook process + connection."""
    srv = next((s for s in cfg.mcp_servers if s.id == server_id), None)
    if srv is None:
        return
    if srv.enabled:
        reset_circuit_breaker(server_id)
        await mcp_mgr.ensure_process(srv)
        await mcp_mgr.connect(srv, skip_process_start=True)
    else:
        await mcp_mgr.stop_process(server_id)
        await mcp_mgr.disconnect(server_id)
    await session_mgr.refresh_tools(cfg)


async def _sync_openclaw_watcher(cfg: AppConfig) -> None:
    """Start or stop the OpenClaw session watcher based on current config."""
    from backend.openclaw_watcher import oc_watcher

    should_run = cfg.openclaw.enabled and cfg.openclaw.watcher_enabled
    if should_run and not oc_watcher.running:
        await oc_watcher.start(poll_interval=cfg.openclaw.watcher_poll_interval)
    elif not should_run and oc_watcher.running:
        await oc_watcher.stop()


@router.post("/settings/openclaw/test")
async def test_openclaw_connection():
    """Test the OpenClaw connection using the current saved config.

    For SSH mode, establishes a connection and lists agents.
    For local mode, checks that the state directory exists.
    Returns agent count on success, error message on failure.
    """
    try:
        cfg = await AppConfig.aload()
        oc = cfg.openclaw
        if not oc.enabled:
            return {"success": False, "message": "OpenClaw integration is not enabled."}

        from tools.transcripts.parsers.openclaw import OpenClawParser
        kwargs: dict = {"mode": oc.mode, "state_dir": oc.state_dir}
        if oc.mode == "ssh":
            if not oc.ssh_host:
                return {"success": False, "message": "SSH host is required."}
            kwargs.update(
                ssh_host=oc.ssh_host,
                ssh_user=oc.ssh_user,
                ssh_key_path=oc.ssh_key_path,
                ssh_port=oc.ssh_port,
            )

        parser = OpenClawParser(**kwargs)
        projects = await asyncio.to_thread(parser.list_projects)
        agent_count = len(projects)
        mode_label = "SSH" if oc.mode == "ssh" else "local"
        return {
            "success": True,
            "message": f"Connected ({mode_label}) — {agent_count} agent{'s' if agent_count != 1 else ''} found.",
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


@router.get("/settings/first-run")
async def first_run():
    """Whether the first-run setup wizard should be presented.

    Resolves to ``True`` when no config exists OR when the user has
    neither completed the wizard nor explicitly dismissed it.  See
    :meth:`AppConfig.is_first_run` for the full predicate.
    """
    cfg = await AppConfig.aload()
    return {
        "first_run": cfg.is_first_run(),
        "completed": cfg.setup.completed,
        "dismissed": cfg.setup.dismissed,
        "current_step": cfg.setup.current_step,
        "completed_steps": list(cfg.setup.completed_steps),
    }


async def _resolve_bedrock_creds(req: TestConnectionRequest) -> dict:
    """Return explicit AWS credential kwargs for boto3/langchain."""
    if req.aws_access_key_id and req.aws_secret_access_key:
        return {
            "aws_access_key_id": req.aws_access_key_id,
            "aws_secret_access_key": req.aws_secret_access_key,
        }
    return {}


def _is_bedrock(req: TestConnectionRequest) -> bool:
    """The frontend may send either "bedrock" or "anthropic_bedrock"
    depending on which code path saved the setting last; treat both as
    Bedrock."""
    return req.model_provider in ("bedrock", "anthropic_bedrock")


@router.post("/settings/test-connection")
async def test_llm_connection(req: TestConnectionRequest):
    try:
        if req.provider == "mlx":
            model_id = (req.hf_llm_model_id or "").strip()
            if not model_id:
                return TestConnectionResponse(success=False, message="HF LLM model ID is required (e.g. mlx-community/quantized-gemma-2b-it).")
            try:
                from huggingface_hub import model_info

                token = (req.hf_token or "").strip() or None
                await asyncio.to_thread(lambda: model_info(model_id, token=token))
            except Exception as exc:
                return TestConnectionResponse(
                    success=False,
                    message=f"Hugging Face Hub check failed: {exc}",
                )
            return TestConnectionResponse(
                success=True,
                message="Model ID is reachable on Hugging Face Hub (MLX loads it locally on first use).",
            )
        if req.provider == "openai":
            if req.openai_model_provider == "azure":
                if not req.azure_endpoint:
                    return TestConnectionResponse(
                        success=False,
                        message="Azure endpoint URL is required for Azure OpenAI.",
                    )
                from langchain_openai import AzureChatOpenAI

                model_name_az = req.model_name or "gpt-4o"
                deployment = req.azure_deployment or model_name_az
                llm = AzureChatOpenAI(
                    azure_endpoint=req.azure_endpoint,
                    azure_deployment=deployment,
                    api_version=req.azure_api_version or "2024-12-01-preview",
                    api_key=req.api_key or "no-key",
                    model=model_name_az,
                    temperature=_openai_temperature(model_name_az),
                    timeout=30.0,
                )
            else:
                if not req.api_key:
                    return TestConnectionResponse(
                        success=False,
                        message="OpenAI API key is required.",
                    )
                from langchain_openai import ChatOpenAI

                model_name_oa = req.model_name or "gpt-4o"
                llm = ChatOpenAI(
                    model=model_name_oa,
                    api_key=req.api_key,
                    temperature=_openai_temperature(model_name_oa),
                    timeout=30.0,
                )
            from langchain_core.messages import HumanMessage
            await llm.ainvoke([HumanMessage(content="Reply with OK")])
            return TestConnectionResponse(success=True, message="Connected successfully")
        if req.provider == "anthropic":
            if _is_bedrock(req):
                if not (req.aws_access_key_id and req.aws_secret_access_key):
                    return TestConnectionResponse(
                        success=False,
                        message="AWS Bedrock requires an Access Key ID and Secret Access Key.",
                    )
                # Validate credentials by listing inference profiles — this
                # confirms the region, auth, and Bedrock access without
                # invoking any model (so legacy/inactive models don't cause
                # spurious failures here).
                import boto3
                from botocore.config import Config as BotoConfig

                boto_cfg = BotoConfig(request_min_compression_size_bytes=1048576)
                session_kwargs: dict = {
                    "region_name": req.bedrock_region,
                    **(await _resolve_bedrock_creds(req)),
                }
                session = await asyncio.to_thread(boto3.Session, **session_kwargs)
                client = await asyncio.to_thread(session.client, "bedrock", config=boto_cfg)

                def _list_profiles() -> list[dict]:
                    profiles: list[dict] = []
                    token = None
                    while True:
                        kwargs: dict = {"maxResults": 1000, "typeEquals": "SYSTEM_DEFINED"}
                        if token:
                            kwargs["nextToken"] = token
                        resp = client.list_inference_profiles(**kwargs)
                        for p in resp.get("inferenceProfileSummaries", []):
                            if p.get("status") != "ACTIVE":
                                continue
                            pmodels = p.get("models", [])
                            if any("anthropic" in (m.get("modelArn", "") or "") for m in pmodels):
                                profiles.append({
                                    "id": p["inferenceProfileId"],
                                    "name": p.get("inferenceProfileName", p["inferenceProfileId"]),
                                })
                        token = resp.get("nextToken")
                        if not token:
                            break
                    return profiles

                profiles = await asyncio.to_thread(_list_profiles)
                count = len(profiles)
                msg = f"Connected — {count} Anthropic inference profile{'s' if count != 1 else ''} available."
                return TestConnectionResponse(success=True, message=msg, models=profiles)
            else:
                from langchain_anthropic import ChatAnthropic
                from langchain_core.messages import HumanMessage
                llm = ChatAnthropic(
                    model=req.model_name or "claude-sonnet-4-6",
                    anthropic_api_key=req.api_key,
                )
                await llm.ainvoke([HumanMessage(content="Reply with OK")])
                return TestConnectionResponse(success=True, message="Connected successfully")
        if req.provider == "omlx":
            from backend import omlx_provisioner as op
            app_cfg = await AppConfig.aload()
            omlx_cfg = app_cfg.omlx
            status = await op.afetch_status(omlx_cfg)
            if not status.get("reachable"):
                return TestConnectionResponse(
                    success=False,
                    message=f"oMLX server unreachable at {omlx_cfg.effective_base_url}: {status.get('error', 'server offline')}",
                )
            model_id = (req.model_name or "").strip()
            if not model_id:
                model_count = len(status.get("models", []))
                return TestConnectionResponse(
                    success=True,
                    message=f"oMLX server reachable — {model_count} model{'s' if model_count != 1 else ''} registered",
                )
            resolved = await op._resolve_omlx_model_id(omlx_cfg, model_id)
            loaded_ids = [m["id"] for m in (status.get("loaded_models") or status.get("models") or [])]
            all_ids = [m["id"] for m in (status.get("models") or [])]
            if resolved and resolved in loaded_ids:
                return TestConnectionResponse(
                    success=True,
                    message=f"oMLX OK — '{resolved}' is loaded and ready",
                )
            if resolved and resolved in all_ids:
                return TestConnectionResponse(
                    success=True,
                    message=f"oMLX reachable — '{resolved}' is registered (use Load to put it in GPU memory)",
                )
            return TestConnectionResponse(
                success=False,
                message=f"oMLX reachable but '{model_id}' is not registered. Download it via the oMLX setup screen.",
            )
        if req.provider == "exo":
            model_id = (req.model_name or "").strip()
            if not model_id:
                return TestConnectionResponse(
                    success=False,
                    message="No EXO model selected.",
                )
            from backend.exo_provisioner import alist_models
            from backend.settings import load_settings
            cfg = (await load_settings()).exo
            result = await alist_models(cfg, timeout=8.0)
            if not result.get("reachable"):
                return TestConnectionResponse(
                    success=False,
                    message=f"EXO cluster unreachable: {result.get('error', 'cluster offline')}",
                )
            loaded = [
                m["id"] for m in result.get("models", [])
                if m.get("id") == model_id and m.get("loaded")
            ]
            downloaded = [
                m["id"] for m in result.get("models", [])
                if m.get("id") == model_id and m.get("downloaded")
            ]
            if loaded:
                return TestConnectionResponse(
                    success=True,
                    message=f"EXO OK — '{model_id}' is loaded and ready",
                )
            if downloaded:
                return TestConnectionResponse(
                    success=True,
                    message=f"EXO reachable — '{model_id}' is downloaded (use Load to put it in memory)",
                )
            return TestConnectionResponse(
                success=False,
                message=f"EXO reachable but '{model_id}' is not downloaded on the cluster",
            )
        return TestConnectionResponse(success=False, message=f"Unknown provider: {req.provider}")
    except Exception as exc:
        return TestConnectionResponse(success=False, message=str(exc))


@router.post("/settings/list-models")
async def list_available_models(req: TestConnectionRequest):
    """Return available model IDs for the configured provider + credentials."""
    try:
        if req.provider == "mlx":
            mid = (req.hf_llm_model_id or "").strip()
            if not mid:
                return {"models": [], "error": None}
            return {"models": [{"id": mid, "name": mid}], "error": None}

        if req.provider == "openai":
            if req.openai_model_provider == "azure":
                # Azure OpenAI has no public model-listing endpoint so we derive
                # the list from the openai SDK's own ChatModel Literal, which is
                # regenerated from OpenAI's OpenAPI spec on every SDK release and
                # therefore stays current automatically.
                try:
                    from openai.types.shared.chat_model import ChatModel
                    _chat_prefixes = ("gpt-", "o1", "o2", "o3", "o4", "chatgpt-", "codex-")
                    azure_models = [
                        {"id": m, "name": m}
                        for m in ChatModel.__args__  # type: ignore[attr-defined]
                        if any(m.startswith(p) for p in _chat_prefixes)
                    ]
                except Exception:
                    azure_models = []
                return {"models": azure_models, "error": None}
            if not req.api_key:
                return {"models": [], "error": "OpenAI API key is required to list models."}
            import httpx

            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {req.api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                # Keep only chat-capable models (gpt-*, o1-*, o3-*, o4-*)
                chat_prefixes = ("gpt-", "o1", "o3", "o4", "chatgpt-")
                models = [
                    {"id": m["id"], "name": m["id"]}
                    for m in data.get("data", [])
                    if any(m["id"].startswith(p) for p in chat_prefixes)
                ]
                models.sort(key=lambda m: m["name"])
                return {"models": models, "error": None}

        if req.provider == "omlx":
            from backend import omlx_provisioner as op
            app_cfg = await AppConfig.aload()
            omlx_cfg = app_cfg.omlx
            status = await op.afetch_status(omlx_cfg)
            if not status.get("reachable"):
                return {
                    "models": [],
                    "error": f"oMLX server unreachable at {omlx_cfg.effective_base_url}: {status.get('error', 'server offline')}",
                }
            models = [{"id": m["id"], "name": m["id"]} for m in (status.get("models") or [])]
            return {"models": models, "error": None}

        if req.provider != "anthropic":
            return {"models": [], "error": f"Unknown provider: {req.provider}"}

        if _is_bedrock(req):
            import boto3
            from botocore.config import Config as BotoConfig

            boto_cfg = BotoConfig(
                request_min_compression_size_bytes=1048576,
            )
            if not (req.aws_access_key_id and req.aws_secret_access_key):
                return {
                    "models": [],
                    "error": "AWS Bedrock requires an Access Key ID and Secret Access Key.",
                }
            session_kwargs: dict = {"region_name": req.bedrock_region, **(await _resolve_bedrock_creds(req))}
            session = boto3.Session(**session_kwargs)
            client = session.client("bedrock", config=boto_cfg)
            models = []
            paginator_token = None
            while True:
                kwargs: dict = {"maxResults": 1000, "typeEquals": "SYSTEM_DEFINED"}
                if paginator_token:
                    kwargs["nextToken"] = paginator_token
                resp = client.list_inference_profiles(**kwargs)
                for p in resp.get("inferenceProfileSummaries", []):
                    if p.get("status") != "ACTIVE":
                        continue
                    profile_models = p.get("models", [])
                    is_anthropic = any("anthropic" in (m.get("modelArn", "") or "") for m in profile_models)
                    if is_anthropic:
                        models.append({
                            "id": p["inferenceProfileId"],
                            "name": p.get("inferenceProfileName", p["inferenceProfileId"]),
                        })
                paginator_token = resp.get("nextToken")
                if not paginator_token:
                    break
            models.sort(key=lambda m: m["name"])
            return {"models": models}

        import httpx

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": req.api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            models = [
                {"id": m["id"], "name": m.get("display_name", m["id"])}
                for m in data.get("data", [])
            ]
            models.sort(key=lambda m: m["name"])
            return {"models": models}

    except Exception as exc:
        return {"models": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Hook auto-install
# ---------------------------------------------------------------------------

_SYSTEM_HOOKS: dict[str, list[dict]] = {
    "PostToolUse": [{"matcher": "*", "hooks": [{"type": "http", "url": "http://localhost:18081/hooks/claude/post-tool-use", "timeout": 10}]}],
    "PostToolUseFailure": [{"matcher": "*", "hooks": [{"type": "http", "url": "http://localhost:18081/hooks/claude/post-tool-use-failure", "timeout": 10}]}],
    "Stop": [{"hooks": [{"type": "http", "url": "http://localhost:18081/hooks/claude/stop", "timeout": 10}]}],
    "SubagentStop": [{"hooks": [{"type": "http", "url": "http://localhost:18081/hooks/claude/subagent-stop", "timeout": 10}]}],
    "SessionStart": [{"hooks": [{"type": "command", "command": "curl -sfS -X POST http://localhost:18081/hooks/claude/session-start -H 'Content-Type: application/json' -d \"$(jq -c '.')\" 2>/dev/null || true", "timeout": 10}]}],
    "SessionEnd": [{"hooks": [{"type": "http", "url": "http://localhost:18081/hooks/claude/session-end", "timeout": 10}]}],
}


def _build_claude_hooks_payload(cfg: AppConfig) -> dict:
    """Build the ``hooks`` object for ``~/.claude/settings.json``."""
    groups: dict[str, list[dict]] = {}

    if cfg.claude_hook.http_hooks_enabled:
        for event, defs in _SYSTEM_HOOKS.items():
            groups[event] = [dict(d) for d in defs]

    for hook in cfg.claude_hook.hooks:
        if not hook.enabled:
            continue
        groups.setdefault(hook.event, [])
        hook_def: dict = {"type": hook.type, "timeout": hook.timeout}
        if hook.type == "http":
            hook_def["url"] = hook.url
        elif hook.type == "command":
            hook_def["command"] = hook.command
        elif hook.type == "prompt":
            hook_def["prompt"] = hook.prompt
        group: dict = {"hooks": [hook_def]}
        if hook.matcher:
            group["matcher"] = hook.matcher
        groups[hook.event].append(group)

    return groups


@router.post("/settings/claude-hooks/install")
async def install_claude_hooks():
    """Merge hooks into ``~/.claude/settings.json``.

    Reads the existing file (or creates it), updates only the ``hooks``
    key, and writes it back — preserving all other Claude Code settings.
    """
    try:
        cfg = await AppConfig.aload()
        hooks_payload = _build_claude_hooks_payload(cfg)

        settings_path = Path.home() / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        existing: dict = {}
        if settings_path.is_file():
            try:
                existing = json.loads(settings_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                existing = {}

        if hooks_payload:
            existing["hooks"] = hooks_payload
        else:
            existing.pop("hooks", None)

        await asyncio.to_thread(
            settings_path.write_text,
            json.dumps(existing, indent=2) + "\n",
            "utf-8",
        )

        hook_count = sum(len(v) for v in hooks_payload.values())
        return {
            "status": "ok",
            "message": f"Installed {hook_count} hook(s) across {len(hooks_payload)} event(s) to {settings_path}",
            "path": str(settings_path),
        }
    except Exception as exc:
        logger.exception("Failed to install Claude hooks")
        return {"status": "error", "message": str(exc)}


@router.post("/shutdown")
async def shutdown():
    """Gracefully shut down the backend server.

    Sends SIGTERM to ourselves so uvicorn runs the FastAPI lifespan
    shutdown (session cleanup, MCP disconnect, scheduler stop) before
    the process exits.
    """
    asyncio.get_running_loop().call_later(0.5, os.kill, os.getpid(), signal.SIGTERM)
    return {"status": "shutting_down"}
