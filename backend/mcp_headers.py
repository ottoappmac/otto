"""HTTP MCP auth headers — Cursor/Claude ``mcpServers`` import + connect.

Remote MCP servers (Snowflake, hosted vendor endpoints) authenticate with
request headers, typically ``Authorization: Bearer <token>``.  Otto's
stdio path already hydrates ``required_secrets`` into subprocess env;
this module is the HTTP equivalent: templates like
``Bearer ${SNOWFLAKE_PAT_TOKEN}`` are expanded from the credential vault
at connect time.  Literal tokens in imported JSON are extracted into
the vault and replaced with a ``${...}`` template so they never persist
in ``config.json``.
"""

from __future__ import annotations

import re
from typing import Any

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_BEARER = re.compile(r"^(Bearer)\s+(\S.*)$", re.IGNORECASE)


class MissingHeaderSecret(KeyError):
    """A ``${NAME}`` header template referenced a vault entry that is absent."""


def shouty_id(server_id: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", server_id.upper()).strip("_")


def header_secret_names(templates: dict[str, str]) -> list[str]:
    """Env-style names referenced by ``${...}`` in header templates."""
    names: list[str] = []
    seen: set[str] = set()
    for value in templates.values():
        for name in _ENV_REF.findall(value or ""):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def ingest_http_headers(
    server_id: str, spec: dict[str, Any],
) -> tuple[dict[str, str], list[str], dict[str, str]]:
    """Parse a Cursor-style ``headers`` object from an MCP JSON spec.

    Returns ``(templates, required_secret_names, vault_values)``.
    ``vault_values`` are literals that must be stored in the keychain;
    ``templates`` never contain those literals.
    """
    raw = spec.get("headers")
    if not isinstance(raw, dict) or not raw:
        return {}, [], {}

    templates: dict[str, str] = {}
    secret_names: list[str] = []
    vault_values: dict[str, str] = {}

    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value:
            continue
        refs = _ENV_REF.findall(value)
        if refs:
            templates[key] = value
            for name in refs:
                if name not in secret_names:
                    secret_names.append(name)
            continue

        bearer = _BEARER.match(value.strip())
        if bearer:
            secret_name = f"{shouty_id(server_id)}_PAT_TOKEN"
            templates[key] = f"{bearer.group(1)} ${{{secret_name}}}"
            if secret_name not in secret_names:
                secret_names.append(secret_name)
            vault_values[secret_name] = bearer.group(2).strip()
            continue

        secret_name = f"{shouty_id(server_id)}_{shouty_id(key)}"
        templates[key] = f"${{{secret_name}}}"
        if secret_name not in secret_names:
            secret_names.append(secret_name)
        vault_values[secret_name] = value

    return templates, secret_names, vault_values


def expand_header_templates(
    templates: dict[str, str], secrets: dict[str, str],
) -> dict[str, str]:
    """Replace ``${NAME}`` placeholders with vault values.

    Raises :class:`MissingHeaderSecret` if a referenced name is absent
    or empty — callers should fail the connect rather than send a
    half-built Authorization header (which Snowflake answers with 401).
    """
    out: dict[str, str] = {}
    for key, tmpl in templates.items():
        def _repl(match: re.Match[str], _secrets: dict[str, str] = secrets) -> str:
            name = match.group(1)
            val = _secrets.get(name) or ""
            if not val:
                raise MissingHeaderSecret(name)
            return val
        out[key] = _ENV_REF.sub(_repl, tmpl)
    return out


def fallback_auth_header(
    header_name: str,
    token_prefix: str,
    required_secrets: list[str],
    secrets: dict[str, str],
) -> dict[str, str] | None:
    """Build a single auth header from the first hydrated required secret.

    Used when a server was registered with ``required_secrets`` but no
    ``headers`` templates (the Snowflake entry already in config.json).
    """
    if not header_name or not required_secrets:
        return None
    for name in required_secrets:
        token = (secrets.get(name) or "").strip()
        if not token:
            continue
        prefix = token_prefix or ""
        if prefix and token.lower().startswith(prefix.lower().rstrip()):
            value = token
        else:
            value = f"{prefix}{token}" if prefix else token
        return {header_name: value}
    return None


def store_ingested_secrets(server_id: str, vault_values: dict[str, str]) -> None:
    """Write extracted header literals into the vault.  Never logs values."""
    if not vault_values:
        return
    from backend.credential_vault import vault

    for name, value in vault_values.items():
        vault.set(server_id, name, value)
