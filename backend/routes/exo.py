"""REST API for the ``exo`` distributed-inference cluster.

Wraps :mod:`backend.exo_provisioner` (which in turn drives
``backend.exo_cli``).  The frontend (``app/src/pages/ExoPage.tsx``)
consumes these endpoints; the deep agent uses :mod:`backend.exo_tools`
which talks to the same provisioner directly.

Endpoints
---------

``GET    /api/exo``                          full info snapshot (paths, prereqs, state)
``GET    /api/exo/status``                   live cluster state via ``/state`` + ``/node_id``
``POST   /api/exo/provision``                kick off provision job (async)
``POST   /api/exo/up``                       provision (if needed) + start daemon
``POST   /api/exo/down``                     stop the local daemon
``POST   /api/exo/smoke``                    run smoke tests against the cluster
``GET    /api/exo/jobs``                     list recent jobs
``GET    /api/exo/jobs/{id}``                poll one job
``GET    /api/exo/log``                      tail of ``exo.log``
``GET    /api/exo/remotes``                  list configured remotes
``POST   /api/exo/remotes``                  add a remote (ssh alias)
``PUT    /api/exo/remotes/{alias}``          update a remote
``DELETE /api/exo/remotes/{alias}``          remove a remote
``POST   /api/exo/remotes/{alias}/up``       up on remote
``POST   /api/exo/remotes/{alias}/down``     down on remote
``POST   /api/exo/remotes/{alias}/smoke``    smoke on remote
``GET    /api/exo/remotes/{alias}/status``   status (cached job result) on remote
``GET    /api/exo/discover/ssh-config``      candidate aliases from ``~/.ssh/config``
``GET    /api/exo/discover/lan``             ``_ssh._tcp`` Bonjour scan of the LAN
``GET    /api/exo/discover/test-ssh``        non-interactive ssh probe of an alias
``GET    /api/exo/discover/tb-link``         live Thunderbolt-Bridge link snapshot
``GET    /api/exo/setup/local-user``         convenience prefill for the wizard's host form
``POST   /api/exo/setup/probe``              probe ``user@host:port`` (no password)
``GET    /api/exo/setup/keypairs``           list candidate local keypairs (no key bytes)
``POST   /api/exo/setup/keypairs``           generate a fresh ED25519 keypair
``POST   /api/exo/setup/install-pubkey``     one-shot: append local pubkey to remote auth keys
``POST   /api/exo/setup/ssh-config``         append a ``Host`` block to ``~/.ssh/config``
``GET    /api/exo/models``                   merged catalog (downloaded + loaded flags)
``POST   /api/exo/models/preload``           warm-load a model (synchronous, deprecated)
``GET    /api/exo/catalog``                  cluster-aware fit-scored catalog
``POST   /api/exo/preload``                  start async preload (returns job_id)
``GET    /api/exo/preload/{id}``             poll a preload job
``GET    /api/exo/preloads``                 list active + recent preload jobs
``POST   /api/exo/preload/{id}/cancel``      cancel a running preload

Setup endpoints
~~~~~~~~~~~~~~~

The ``/setup/*`` endpoints back the Cluster setup wizard in
``app/src/pages/SettingsPage.tsx``. They are *strictly* human-driven:
nothing in :mod:`backend.exo_tools` exposes them to the LLM. The only
endpoint that ever accepts a password is ``/setup/install-pubkey``;
that route uses a Pydantic ``SecretStr`` for the field, never echoes
the body in the response, and discards the value as soon as
``asyncssh`` returns. See :mod:`backend.exo_setup` for the threat-model
notes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, SecretStr

from backend import exo_provisioner as ep
from backend import exo_discovery
from backend import exo_setup
from backend.config import AppConfig, ExoConfig, ExoRemoteConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/exo", tags=["exo"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _load_cfg() -> tuple[AppConfig, ExoConfig]:
    app_cfg = await AppConfig.aload()
    return app_cfg, app_cfg.exo


async def _save_exo_cfg(app_cfg: AppConfig, exo_cfg: ExoConfig) -> None:
    app_cfg.exo = exo_cfg
    await app_cfg.asave()
    app_cfg.apply_to_environ()


def _find_remote(cfg: ExoConfig, alias: str) -> ExoRemoteConfig:
    for r in cfg.remotes:
        if r.ssh_alias == alias:
            return r
    raise HTTPException(status_code=404, detail=f"Remote '{alias}' not found")


# ---------------------------------------------------------------------------
# Snapshots / read-only
# ---------------------------------------------------------------------------


@router.get("")
async def exo_info() -> dict:
    _, cfg = await _load_cfg()
    return ep.info_snapshot(cfg)


@router.get("/status")
async def exo_status() -> dict:
    _, cfg = await _load_cfg()
    return await ep.afetch_status(cfg)


@router.get("/log")
async def exo_log(lines: int = 200) -> dict:
    return {"lines": ep.tail_log(max_lines=max(1, min(2000, lines)))}


# ---------------------------------------------------------------------------
# Local lifecycle
# ---------------------------------------------------------------------------


class _ProvisionRequest(BaseModel):
    force: bool = False
    force_mismatch: bool = False


@router.post("/provision")
async def exo_provision(req: _ProvisionRequest) -> dict:
    _, cfg = await _load_cfg()
    if not cfg.enabled:
        cfg.enabled = True  # implicit opt-in on first explicit provision
    job = await ep.aprovision(cfg, force=req.force, force_mismatch=req.force_mismatch)
    return {"job_id": job.id, **job.to_dict()}


@router.post("/up")
async def exo_up(req: _ProvisionRequest) -> dict:
    _, cfg = await _load_cfg()
    job = await ep.aup_job(cfg, force=req.force, force_mismatch=req.force_mismatch)
    return {"job_id": job.id, **job.to_dict()}


@router.get("/release/check")
async def exo_release_check() -> dict:
    """Probe GitHub for the latest EXO release and compare to the local pin.

    No-op when the host has no network or GitHub returns an unexpected
    payload — we surface a structured ``{"ok": False, "error": ...}``
    response instead of raising so the UI can render a one-liner without
    blowing up the EXO setup screen.

    Response:
        {
          "ok": true,
          "current_ref": "v1.0.71",
          "latest_tag": "v1.0.85",
          "newer_available": true,
          "html_url": "https://github.com/exo-explore/exo/releases/tag/v1.0.85",
          "published_at": "2026-04-12T18:34:21Z",
        }
    """
    import json
    import urllib.error
    import urllib.request

    from backend.exo_cli import DEFAULT_EXO_REF, load_state

    _, cfg = await _load_cfg()
    current_ref = (cfg.repo_ref or load_state().exo_ref or DEFAULT_EXO_REF).strip()

    # Derive ``owner/repo`` from the configured repo_url so a forked
    # mirror still gets queried correctly.  Falls back to the canonical
    # exo-explore/exo if parsing fails.
    repo_url = (cfg.repo_url or "https://github.com/exo-explore/exo.git").strip()
    owner_repo = "exo-explore/exo"
    try:
        path = repo_url.split("github.com/", 1)[1]
        owner_repo = path.removesuffix(".git").strip("/") or owner_repo
    except IndexError:
        pass

    api_url = f"https://api.github.com/repos/{owner_repo}/releases/latest"
    req = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "otto-exo-release-check",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "current_ref": current_ref,
            "error": f"GitHub returned {exc.code}",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "current_ref": current_ref,
            "error": f"Network error: {exc}",
        }
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "ok": False,
            "current_ref": current_ref,
            "error": f"Could not parse GitHub response: {exc}",
        }

    latest = (payload.get("tag_name") or "").strip()
    return {
        "ok": True,
        "current_ref": current_ref,
        "latest_tag": latest,
        # Simple string mismatch — we deliberately don't try to do
        # semver-style "is X newer than Y" comparisons because EXO has
        # used a mix of ``v1.0.x`` and date-based tags, and a string
        # compare is honest about what it is: "your pin doesn't match
        # the published latest, here's the link, go decide."
        "newer_available": bool(latest) and latest != current_ref,
        "html_url": payload.get("html_url") or f"https://github.com/{owner_repo}/releases",
        "published_at": payload.get("published_at") or "",
    }


@router.post("/down")
async def exo_down() -> dict:
    _, cfg = await _load_cfg()
    try:
        return await ep.adown_local(cfg)
    except ep.ExoCliError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/smoke")
async def exo_smoke() -> dict:
    _, cfg = await _load_cfg()
    job = await ep.asmoke(cfg)
    return {"job_id": job.id, **job.to_dict()}


# ---------------------------------------------------------------------------
# Job polling
# ---------------------------------------------------------------------------


@router.get("/jobs")
async def list_exo_jobs() -> dict:
    return {"jobs": [j.to_dict() for j in ep.list_jobs()]}


@router.get("/jobs/{job_id}")
async def get_exo_job(job_id: str) -> dict:
    job = ep.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.to_dict()


# ---------------------------------------------------------------------------
# Remotes
# ---------------------------------------------------------------------------


class _RemoteCreate(BaseModel):
    ssh_alias: str = Field(..., min_length=1)
    label: str = ""
    app_data_dir: str = ""
    enabled: bool = True


class _RemoteUpdate(BaseModel):
    label: Optional[str] = None
    app_data_dir: Optional[str] = None
    enabled: Optional[bool] = None


@router.get("/remotes")
async def list_remotes() -> dict:
    _, cfg = await _load_cfg()
    return {"remotes": [r.model_dump() for r in cfg.remotes]}


@router.post("/remotes")
async def add_remote(req: _RemoteCreate) -> dict:
    app_cfg, cfg = await _load_cfg()
    if any(r.ssh_alias == req.ssh_alias for r in cfg.remotes):
        raise HTTPException(
            status_code=409,
            detail=f"Remote '{req.ssh_alias}' already exists",
        )
    cfg.remotes.append(ExoRemoteConfig(**req.model_dump()))
    await _save_exo_cfg(app_cfg, cfg)
    return {"status": "saved", "remotes": [r.model_dump() for r in cfg.remotes]}


@router.put("/remotes/{alias}")
async def update_remote(alias: str, req: _RemoteUpdate) -> dict:
    app_cfg, cfg = await _load_cfg()
    remote = _find_remote(cfg, alias)
    data = req.model_dump(exclude_none=True)
    for k, v in data.items():
        setattr(remote, k, v)
    await _save_exo_cfg(app_cfg, cfg)
    return {"status": "saved", "remote": remote.model_dump()}


@router.delete("/remotes/{alias}")
async def remove_remote(alias: str) -> dict:
    app_cfg, cfg = await _load_cfg()
    before = len(cfg.remotes)
    cfg.remotes = [r for r in cfg.remotes if r.ssh_alias != alias]
    if len(cfg.remotes) == before:
        raise HTTPException(status_code=404, detail=f"Remote '{alias}' not found")
    await _save_exo_cfg(app_cfg, cfg)
    return {"status": "removed", "remaining": len(cfg.remotes)}


@router.post("/remotes/{alias}/up")
async def remote_up(alias: str, req: _ProvisionRequest) -> dict:
    _, cfg = await _load_cfg()
    remote = _find_remote(cfg, alias)
    args = ep._remote_subcmd_for("up", cfg)
    if req.force:
        args.append("--force")
    job = await ep.arun_remote(cfg, remote, args, job_kind="up")
    return {"job_id": job.id, **job.to_dict()}


@router.post("/remotes/{alias}/down")
async def remote_down(alias: str) -> dict:
    _, cfg = await _load_cfg()
    remote = _find_remote(cfg, alias)
    job = await ep.arun_remote(cfg, remote, ep._remote_subcmd_for("down", cfg), job_kind="down")
    return {"job_id": job.id, **job.to_dict()}


@router.post("/remotes/{alias}/smoke")
async def remote_smoke(alias: str) -> dict:
    _, cfg = await _load_cfg()
    remote = _find_remote(cfg, alias)
    job = await ep.arun_remote(cfg, remote, ep._remote_subcmd_for("smoke", cfg), job_kind="smoke")
    return {"job_id": job.id, **job.to_dict()}


@router.get("/remotes/{alias}/status")
async def remote_status(alias: str) -> dict:
    _, cfg = await _load_cfg()
    remote = _find_remote(cfg, alias)
    job = await ep.arun_remote(cfg, remote, ep._remote_subcmd_for("status", cfg), job_kind="status")
    return {"job_id": job.id, **job.to_dict()}


# ---------------------------------------------------------------------------
# Discovery — used by the Settings "Add remote" autocomplete + LAN scan.
# Both endpoints are read-only and have no side effects on the cluster.
# ---------------------------------------------------------------------------


@router.get("/discover/ssh-config")
async def discover_ssh_config() -> dict:
    """Parse ``~/.ssh/config`` (and any ``Include`` files) into concrete
    host aliases the UI can offer as autocomplete suggestions."""
    import asyncio
    hosts = await asyncio.to_thread(exo_discovery.ssh_config_to_dicts)
    return {"hosts": hosts}


@router.get("/discover/lan")
async def discover_lan(timeout: float = 3.0) -> dict:
    """Run a brief Bonjour/mDNS browse for ``_ssh._tcp`` services on the
    local network. Cross-references hits against ``~/.ssh/config`` so the
    UI can hint when a discovered host is already a known alias.

    ``timeout`` is clamped to [0.5s, 15s] inside the discovery module.
    """
    import asyncio
    hosts = await asyncio.to_thread(exo_discovery.lan_scan_to_dicts, timeout)
    return {"hosts": hosts, "timeout": timeout}


@router.get("/discover/test-ssh")
async def discover_test_ssh(alias: str, timeout: float = 6.0) -> dict:
    """Run a non-interactive ``ssh <alias> echo …`` probe and return the
    result. Lets the UI verify a candidate alias is reachable with key
    auth *before* committing to provision."""
    import asyncio
    if not alias.strip():
        raise HTTPException(status_code=400, detail="alias is required")
    return await asyncio.to_thread(exo_discovery.test_ssh, alias, timeout=timeout)


@router.get("/discover/tb-link")
async def discover_tb_link() -> dict:
    """Live snapshot of any Thunderbolt-Bridge link.

    The Cluster setup wizard polls this so it can offer one-click
    "Thunderbolt cable detected — peer reachable at 169.254.x.y" hints.
    Returns ``{"connected": False}`` when no TB-Bridge is up.
    """
    import asyncio
    return await asyncio.to_thread(exo_discovery.live_thunderbolt_link)


# ---------------------------------------------------------------------------
# Setup wizard (Settings → LLM → Cluster → "Add remote → Set up new node")
#
# These endpoints are deliberately *not* exposed to the deep agent. The
# only endpoint that handles a password is /setup/install-pubkey; it
# uses Pydantic SecretStr, never echoes the body, and the value never
# enters subprocess argv (asyncssh accepts it as a kwarg).
# ---------------------------------------------------------------------------


class _SetupProbeRequest(BaseModel):
    host: str = Field(..., min_length=1)
    user: str = Field(..., min_length=1)
    port: int = 22
    timeout: float = 6.0


@router.get("/setup/local-user")
async def setup_local_user() -> dict:
    """Convenience prefill: current $USER, ~/.ssh existence + perms."""
    import asyncio
    return await asyncio.to_thread(exo_setup.local_user_info)


@router.post("/setup/probe")
async def setup_probe(req: _SetupProbeRequest) -> dict:
    """Non-interactive probe of ``user@host:port`` (no password).

    Reports TCP reachability, key-auth success, OS/arch, and whether
    ``uv`` and ``exo`` are already installed on the remote so the
    wizard can decide which steps to skip.
    """
    return await exo_setup.probe_host(
        host=req.host, user=req.user, port=req.port, timeout=req.timeout,
    )


@router.get("/setup/keypairs")
async def setup_list_keypairs() -> dict:
    """Enumerate candidate keypairs under ``~/.ssh/``. Never returns key bytes."""
    import asyncio
    keys = await asyncio.to_thread(exo_setup.list_local_keypairs)
    return {"keypairs": keys}


class _CreateKeypairRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    key_type: str = "ed25519"
    comment: str = ""


@router.post("/setup/keypairs")
async def setup_create_keypair(req: _CreateKeypairRequest) -> dict:
    """Generate a fresh keypair under ``~/.ssh/<name>``.

    Idempotent — refuses to overwrite an existing key. Returns
    fingerprint metadata only; the private key stays on disk.
    """
    try:
        return await exo_setup.create_keypair(
            req.name, key_type=req.key_type, comment=req.comment,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class _InstallPubkeyRequest(BaseModel):
    """Body for ``POST /setup/install-pubkey``.

    ``password`` is a :class:`SecretStr` so its repr/json is
    ``**********`` — even if the body lands in a stray log line, the
    cleartext is not exposed.
    """

    host: str = Field(..., min_length=1)
    user: str = Field(..., min_length=1)
    port: int = 22
    password: SecretStr
    public_key_path: str = Field(..., min_length=1)
    private_key_path: str = ""
    timeout: float = 15.0


@router.post("/setup/install-pubkey")
async def setup_install_pubkey(req: _InstallPubkeyRequest) -> dict:
    """One-shot password authentication to install a public key.

    The password lives only between this handler and ``asyncssh.connect``;
    the response never echoes the input. On any error we raise a generic
    500 — the client gets a short, non-revealing message.
    """
    try:
        return await exo_setup.install_authorized_key(
            host=req.host,
            user=req.user,
            port=req.port,
            password=req.password.get_secret_value(),
            public_key_path=req.public_key_path,
            private_key_path=req.private_key_path,
            timeout=req.timeout,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Generic message — never echo the request body. The detail is
        # the type-name + a short reason from exo_setup.install_authorized_key.
        logger.warning("install_authorized_key failed for %s@%s: %s",
                       req.user, req.host, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class _SshConfigAppendRequest(BaseModel):
    alias: str = Field(..., min_length=1)
    hostname: str = Field(..., min_length=1)
    user: str = ""
    port: int = 22
    identity_file: str = ""
    extra_options: dict[str, str] = Field(default_factory=dict)
    replace: bool = False


@router.post("/setup/ssh-config")
async def setup_ssh_config(req: _SshConfigAppendRequest) -> dict:
    """Append a ``Host`` block to ``~/.ssh/config`` (with a backup)."""
    import asyncio
    try:
        return await asyncio.to_thread(
            exo_setup.append_ssh_config_block,
            alias=req.alias,
            hostname=req.hostname,
            user=req.user,
            port=req.port,
            identity_file=req.identity_file,
            extra_options=req.extra_options,
            replace=req.replace,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Model catalog + preload
# ---------------------------------------------------------------------------


@router.get("/models")
async def exo_models() -> dict:
    """Return the cluster's full model catalog with ``downloaded`` and
    ``loaded`` flags merged in. Empty list when the cluster is offline.
    """
    _, cfg = await _load_cfg()
    return await ep.alist_models(cfg)


class _PreloadRequest(BaseModel):
    model: str = Field(..., min_length=1)
    # Optional per-call override.  When omitted, the placement uses
    # ``ExoConfig.min_nodes`` from saved settings.
    min_nodes: Optional[int] = None


@router.post("/models/preload")
async def exo_models_preload(req: _PreloadRequest) -> dict:
    """Warm-load ``model`` on the cluster via ``POST /place_instance``.

    exo will lazily download (if needed) and load the model into memory;
    subsequent requests against the same id will be hot.  When
    ``min_nodes`` (request body) or ``cfg.min_nodes`` (settings) is >1,
    the placement is forced to pipeline-parallel across that many nodes.
    """
    _, cfg = await _load_cfg()
    return await ep.apreload_model(
        cfg, req.model, timeout=600.0, min_nodes=req.min_nodes
    )


# ---------------------------------------------------------------------------
# Catalog + async preload — the "ModelChooser" surface.
# ---------------------------------------------------------------------------


@router.get("/catalog")
async def exo_catalog(
    min_nodes: int = 1,
    ctx_len: int = 8192,
    kv_bits: Optional[int] = None,
    refresh: bool = False,
) -> dict:
    """Return the curated model catalog scored against the live cluster.

    The cards are loaded from
    ``<exo_repo>/resources/inference_model_cards/*.toml`` and joined with:

    * ``downloaded`` / ``loaded`` flags from
      :func:`backend.exo_provisioner.alist_models`
    * Per-node hardware budgets from
      :func:`backend.exo_provisioner.afetch_status`

    The result is sorted with loaded → comfortable → tight → over so
    the UI can render it directly.  Empty (but well-formed) when no
    cluster is reachable; the ``capabilities`` block lets the frontend
    show a fallback hint.
    """
    from backend.exo_catalog import get_catalog, score_catalog

    _, cfg = await _load_cfg()

    # Cards live on disk regardless of whether the cluster is up. The
    # location differs by delivery mode (prebuilt runtime vs source repo),
    # so resolve it via the mode-aware helper.
    cards_dir = ep.model_cards_dir(cfg)
    rows = get_catalog(cards_dir, force=refresh)

    # Live overlay — cluster status (for nodes) and model list (for
    # downloaded/loaded flags).  Both can fail when the cluster is
    # down; we degrade gracefully.
    nodes: list = []
    peer_count = 0
    reachable = False
    cluster_error: Optional[str] = None
    try:
        status = await ep.afetch_status(cfg)
        reachable = bool(status.get("reachable"))
        peer_count = int(status.get("peer_count") or 0)
        nodes = [
            {
                "node_id": n.get("node_id"),
                "chip": n.get("chip"),
                "memory_total_gb": n.get("memory_total_gb"),
                "memory_free_gb": n.get("memory_free_gb"),
            }
            for n in (status.get("nodes") or [])
        ]
        if not reachable:
            cluster_error = status.get("error")
    except Exception as exc:  # noqa: BLE001
        cluster_error = str(exc)

    downloaded: set[str] = set()
    loaded: set[str] = set()
    try:
        models = await ep.alist_models(cfg)
        for m in models.get("models") or []:
            mid = str(m.get("id") or "")
            if not mid:
                continue
            if m.get("downloaded"):
                downloaded.add(mid)
            if m.get("loaded"):
                loaded.add(mid)
    except Exception as exc:  # noqa: BLE001
        if cluster_error is None:
            cluster_error = str(exc)

    # Apply downloaded/loaded overlay before scoring so the sort can
    # surface loaded models to the top.
    for r in rows:
        r.downloaded = r.model_id in downloaded
        r.loaded = r.model_id in loaded

    # Cap min_nodes at peer count so the slider can't exceed reality.
    capped_min_nodes = max(1, min(int(min_nodes or 1), max(peer_count, 1)))

    scored = score_catalog(
        rows,
        nodes=nodes,
        min_nodes=capped_min_nodes,
        ctx_len=int(ctx_len or 8192),
        kv_bits=kv_bits,
    )

    counts = {
        "total": len(scored),
        "comfortable": sum(1 for r in scored if r.get("fits") == "comfortable"),
        "tight": sum(1 for r in scored if r.get("fits") == "tight"),
        "over": sum(1 for r in scored if r.get("fits") == "over"),
        "downloaded": sum(1 for r in scored if r.get("downloaded")),
        "loaded": sum(1 for r in scored if r.get("loaded")),
    }

    return {
        "rows": scored,
        "counts": counts,
        "cluster": {
            "reachable": reachable,
            "peer_count": peer_count,
            "min_nodes": capped_min_nodes,
            "max_nodes": max(peer_count, 1),
            "nodes": nodes,
            "error": cluster_error,
        },
        "params": {
            "ctx_len": int(ctx_len or 8192),
            "kv_bits": kv_bits,
        },
    }


class _PreloadStartRequest(BaseModel):
    model: str = Field(..., min_length=1)
    min_nodes: Optional[int] = Field(default=None, ge=1)
    timeout: float = Field(default=1800.0, gt=0)


@router.post("/preload")
async def exo_preload_start(req: _PreloadStartRequest) -> dict:
    """Spawn an async preload job, returning the initial public view.

    Unlike :func:`exo_models_preload` (which blocks the HTTP request
    until exo has fully placed the model — fine for CLI-style callers
    but bad for a UI), this endpoint returns immediately with a
    ``job_id`` the frontend can poll for byte/file progress.
    """
    _, cfg = await _load_cfg()
    if not ep.is_running(cfg.api_port):
        raise HTTPException(
            status_code=409,
            detail="exo daemon is not running locally — start it first",
        )
    snap = ep.start_preload_job(
        cfg, req.model, timeout=req.timeout, min_nodes=req.min_nodes
    )
    return snap


@router.get("/preload/{job_id}")
async def exo_preload_status(job_id: str) -> dict:
    snap = ep.get_preload_job(job_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"unknown preload job {job_id!r}")
    return snap


@router.get("/preloads")
async def exo_preload_list() -> dict:
    return {"jobs": ep.list_preload_jobs()}


@router.post("/preload/{job_id}/cancel")
async def exo_preload_cancel(job_id: str) -> dict:
    ok = ep.cancel_preload_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"unknown preload job {job_id!r}")
    return {"status": "cancelled", "job_id": job_id}


class _UnloadRequest(BaseModel):
    model: str


@router.post("/unload")
async def exo_unload(req: _UnloadRequest) -> dict:
    """Delete all in-memory instances for *model*, freeing cluster RAM.

    Calls ``DELETE /instance/{id}`` on every ``/state.instances`` entry
    whose ``modelId`` matches the requested model.  Returns the number of
    instances removed.  Does **not** remove downloaded weights from disk.
    """
    _, cfg = await _load_cfg()
    if not ep.is_running(cfg.api_port):
        raise HTTPException(
            status_code=409,
            detail="exo daemon is not running locally — start it first",
        )
    base = cfg.effective_base_url.rstrip("/")
    deleted = await asyncio.to_thread(
        ep._delete_instances_for_model_sync, base, req.model
    )
    return {"status": "unloaded", "model": req.model, "instances_removed": deleted}
