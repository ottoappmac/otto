"""LangChain tools that let the chat agent manage agents, skills, and discover MCP tools."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from backend.config import AppConfig
from backend.schemas import AgentSpec, SkillSpec
from backend.utils import run_coro_sync

logger = logging.getLogger(__name__)


async def _refresh_active_sessions() -> None:
    """Rebuild the agent graph for every active session so newly-saved
    agent configs become visible without forcing the user to restart
    the session.

    Best-effort: failures are logged but do not propagate, because the
    create/update operation that triggered this has already succeeded
    on disk.  The next time a session is built (new chat, resume, or
    explicit refresh) it will pick up the change either way.
    """
    try:
        from backend.state import session_mgr

        cfg = await AppConfig.aload()
        await session_mgr.refresh_tools(cfg)
    except Exception:
        logger.debug(
            "refresh_tools failed after agent-config change", exc_info=True,
        )


def build_management_tools(mcp_mgr: Any) -> list:
    """Build management tools that close over the given MCPManager instance.

    Args:
        mcp_mgr: The MCPManager for the current session (used to introspect connected tools).

    Returns:
        List of LangChain tool callables to inject into the deep agent.
    """

    @tool
    def discover_available_tools(server_id: str = "") -> str:
        """Discover MCP tools available in the system with their names, descriptions,
        and input parameter schemas. Use this before creating skills to understand
        what tools exist and how they work.

        Args:
            server_id: Optional MCP server ID to filter by (e.g. "playwright-mcp").
                       Leave empty to return tools from all connected servers.
        """
        results = []
        for sid, conn in mcp_mgr.connections.items():
            if server_id and sid != server_id:
                continue
            if not conn.connected:
                continue
            for t in conn.tools:
                try:
                    schema = t.get_input_jsonschema()
                except Exception:
                    schema = {}
                results.append({
                    "server": sid,
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": schema,
                })
        if not results:
            hint = f" (filter: {server_id})" if server_id else ""
            return f"No connected tools found.{hint}"
        return json.dumps(results, indent=2)

    @tool
    def list_available_mcp_servers() -> str:
        """List all configured MCP tool servers and their connection status.
        Returns server IDs, names, transport type, and how many tools each has."""
        cfg = AppConfig.load()
        lines = []
        for srv in cfg.mcp_servers:
            conn = mcp_mgr.connections.get(srv.id)
            status = "connected" if (conn and conn.connected) else "disconnected"
            tool_count = len(conn.tools) if conn and conn.connected else 0
            lines.append(
                f"- **{srv.id}** ({srv.name}): {srv.transport}, {status}, {tool_count} tools"
            )
        return "\n".join(lines) or "No MCP servers configured."

    @tool
    def list_existing_skills() -> str:
        """List all existing skills with their names and descriptions.
        Use this to see what skills already exist before creating new ones."""
        from backend.agent_library import list_skills

        skills = list_skills()
        if not skills:
            return "No skills configured."
        lines = []
        for s in skills:
            lines.append(f"- **{s.name}**: {s.description}")
        return "\n".join(lines)

    @tool
    def list_existing_agents() -> str:
        """List all existing agents with their names, descriptions, tools, and skills.
        Use this to see what agents already exist before creating new ones."""
        from backend.agent_library import list_agents

        agents = list_agents()
        if not agents:
            return "No agents configured."
        lines = []
        for a in agents:
            lines.append(f"- **{a.name}**: {a.description}")
            lines.append(f"  Tools: {a.tools}, Skills: {a.skills}")
        return "\n".join(lines)

    @tool
    def create_skill(
        name: str,
        description: str,
        content: str,
    ) -> str:
        """Create a new skill definition that teaches agents how to use specific tools.

        The skill content should be a complete SKILL.md document with YAML frontmatter
        and markdown body. Use discover_available_tools first to understand the tools,
        then write a skill that documents the workflow and rules for using them.

        Args:
            name: Kebab-case skill name (e.g. "salesforce-navigation")
            description: One-line description of when this skill should be used
            content: Full SKILL.md content including YAML frontmatter and markdown body.
                     Must include sections: When to use, Core Workflow, Critical Rules, Prerequisites.
                     Example frontmatter:
                     ---
                     name: my-skill
                     description: When to use this skill
                     ---
        """
        from backend.agent_library import get_skill, save_skill

        if get_skill(name):
            return f"Error: Skill '{name}' already exists. Choose a different name or delete the existing one first."

        spec = SkillSpec(
            name=name,
            description=description,
            content=content,
        )
        saved = save_skill(spec)
        return f"Skill '{saved.name}' created successfully."

    @tool
    def create_agent_config(
        name: str,
        description: str,
        system_prompt: str,
        tools: list[str],
        skills: list[str],
        subagent_llm_family: str | None = None,
        model_override: str | None = None,
        mlx_model_id: str | None = None,
    ) -> str:
        """Create a new agent configuration that combines MCP tool servers with skills.

        The agent will appear in the Agents page and can be selected when starting
        a new chat session.

        Args:
            name: Kebab-case agent name (e.g. "sap-order-creator")
            description: 1-2 sentence summary of what the agent does
            system_prompt: Complete system instructions in markdown. Should include sections:
                           Role, Core Workflow, Key Rules, Speed, Output Conventions.
            tools: List of MCP server IDs to attach (e.g. ["playwright-mcp"]).
                   Use list_available_mcp_servers to find valid IDs.
            skills: List of skill names to attach (e.g. ["playwright-browser"]).
                    Use list_existing_skills to find valid names. Create skills first if needed.
            subagent_llm_family: LLM stack for this agent when invoked as a subagent.
                One of: "inherit" (use the parent session's model), "frontier" (Anthropic API),
                "mlx" (local MLX; optionally specify mlx_model_id), "exo" (distributed exo
                cluster), "custom" (arbitrary model via model_override). Leave None to inherit.
            model_override: Model identifier used when subagent_llm_family is "custom"
                (e.g. "claude-opus-4-5", "gpt-4o"). Ignored for other families.
            mlx_model_id: HuggingFace repo id used when subagent_llm_family is "mlx"
                (e.g. "mlx-community/Qwen3-8B-4bit"). Defaults to the global MLX model
                when omitted. Note: first use may block while weights download.
        """
        from backend.agent_library import get_skill, list_agents, save_agent

        if any(a.name == name for a in list_agents()):
            return f"Error: Agent '{name}' already exists. Choose a different name."

        for skill_name in skills:
            if not get_skill(skill_name):
                return f"Error: Skill '{skill_name}' not found. Create it first with create_skill."

        cfg = AppConfig.load()
        valid_ids = {s.id for s in cfg.mcp_servers}
        for t in tools:
            if t not in valid_ids:
                return f"Error: MCP server '{t}' not found. Available: {sorted(valid_ids)}"

        _VALID_FAMILIES = {"inherit", "frontier", "mlx", "exo", "custom"}
        if subagent_llm_family and subagent_llm_family not in _VALID_FAMILIES:
            return (
                f"Error: subagent_llm_family must be one of {sorted(_VALID_FAMILIES)}, "
                f"got '{subagent_llm_family}'."
            )
        if subagent_llm_family == "custom" and not model_override:
            return "Error: model_override is required when subagent_llm_family is 'custom'."

        spec = AgentSpec(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools,
            skills=skills,
            subagent_llm_family=subagent_llm_family or None,
            model_override=model_override or None,
            mlx_model_id=mlx_model_id or None,
        )
        saved = save_agent(spec)
        # Rebuild active sessions so the new sub-agent is bound on the
        # next user message.  Without this, ``task`` cannot dispatch to
        # the agent we just created until the user closes and reopens
        # the chat.
        run_coro_sync(_refresh_active_sessions())
        model_info = f", LLM: {saved.subagent_llm_family or 'inherit'}" + (
            f" ({saved.model_override})" if saved.model_override else
            f" ({saved.mlx_model_id})" if saved.mlx_model_id else ""
        )
        return (
            f"Agent '{saved.name}' created successfully. "
            f"Tools: {saved.tools}, Skills: {saved.skills}{model_info}. "
            f"Available as a subagent on the next message in this session, "
            f"or call spawn_followup_session to hand off the current request "
            f"to a fresh session that has it bound from the start."
        )

    @tool
    def update_skill(
        name: str,
        description: str = "",
        content: str = "",
    ) -> str:
        """Update an existing skill. Only provided fields are changed; omitted fields keep their current values.

        Args:
            name: Name of the skill to update (must already exist)
            description: New one-line description (leave empty to keep current)
            content: New full SKILL.md content including YAML frontmatter and markdown body (leave empty to keep current)
        """
        from backend.agent_library import get_skill, save_skill

        existing = get_skill(name)
        if not existing:
            return f"Error: Skill '{name}' not found. Use list_existing_skills to see available skills."

        if description:
            existing.description = description
        if content:
            existing.content = content

        saved = save_skill(existing)
        return f"Skill '{saved.name}' updated."

    @tool
    def update_agent_config(
        name: str,
        description: str = "",
        system_prompt: str = "",
        tools: list[str] | None = None,
        skills: list[str] | None = None,
        subagent_llm_family: str | None = None,
        model_override: str | None = None,
        mlx_model_id: str | None = None,
    ) -> str:
        """Update an existing agent configuration. Only provided fields are changed; omitted fields keep their current values.

        Args:
            name: Name of the agent to update (must already exist)
            description: New 1-2 sentence summary (leave empty to keep current)
            system_prompt: New system instructions in markdown (leave empty to keep current)
            tools: New list of MCP server IDs (omit or pass null to keep current).
                   Use list_available_mcp_servers to find valid IDs.
            skills: New list of skill names (omit or pass null to keep current).
                    Use list_existing_skills to find valid names.
            subagent_llm_family: LLM stack for this agent when invoked as a subagent.
                One of: "inherit" (use the parent session's model), "frontier" (Anthropic API),
                "mlx" (local MLX; optionally specify mlx_model_id), "exo" (distributed exo
                cluster), "custom" (arbitrary model via model_override). Omit to keep current.
            model_override: Model identifier used when subagent_llm_family is "custom"
                (e.g. "claude-opus-4-5", "gpt-4o"). Pass empty string to clear.
            mlx_model_id: HuggingFace repo id used when subagent_llm_family is "mlx"
                (e.g. "mlx-community/Qwen3-8B-4bit"). Pass empty string to revert to
                the global MLX model. Note: first use may block while weights download.
        """
        from backend.agent_library import get_agent, get_skill, save_agent

        existing = get_agent(name)
        if not existing:
            return f"Error: Agent '{name}' not found. Use list_existing_agents to see available agents."

        if description:
            existing.description = description
        if system_prompt:
            existing.system_prompt = system_prompt
        if tools is not None:
            cfg = AppConfig.load()
            valid_ids = {s.id for s in cfg.mcp_servers}
            for t in tools:
                if t not in valid_ids:
                    return f"Error: MCP server '{t}' not found. Available: {sorted(valid_ids)}"
            existing.tools = tools
        if skills is not None:
            for skill_name in skills:
                if not get_skill(skill_name):
                    return f"Error: Skill '{skill_name}' not found. Create it first with create_skill."
            existing.skills = skills

        _VALID_FAMILIES = {"inherit", "frontier", "mlx", "exo", "custom"}
        if subagent_llm_family is not None:
            if subagent_llm_family not in _VALID_FAMILIES:
                return (
                    f"Error: subagent_llm_family must be one of {sorted(_VALID_FAMILIES)}, "
                    f"got '{subagent_llm_family}'."
                )
            existing.subagent_llm_family = subagent_llm_family
        if model_override is not None:
            existing.model_override = model_override or None
        if mlx_model_id is not None:
            existing.mlx_model_id = mlx_model_id or None

        # Validate family + override consistency after applying all changes.
        if existing.subagent_llm_family == "custom" and not existing.model_override:
            return "Error: model_override is required when subagent_llm_family is 'custom'."

        saved = save_agent(existing)
        # Rebuild active sessions so updated bindings take effect on the
        # next user message instead of waiting for a session restart.
        run_coro_sync(_refresh_active_sessions())
        model_info = f", LLM: {saved.subagent_llm_family or 'inherit'}" + (
            f" ({saved.model_override})" if saved.model_override else
            f" ({saved.mlx_model_id})" if saved.mlx_model_id else ""
        )
        return (
            f"Agent '{saved.name}' updated. "
            f"Tools: {saved.tools}, Skills: {saved.skills}{model_info}."
        )

    return [
        discover_available_tools,
        list_available_mcp_servers,
        list_existing_skills,
        list_existing_agents,
        create_skill,
        create_agent_config,
        update_skill,
        update_agent_config,
    ]
