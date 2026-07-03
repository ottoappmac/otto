"""Agent and skill management API routes."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.agent_library import (
    delete_agent,
    delete_skill,
    get_agent,
    get_skill,
    is_builtin_agent,
    is_builtin_skill,
    list_agents,
    list_skills,
    save_agent,
    save_skill,
)
from backend.config import AppConfig
from backend.prompts import AGENT_GENERATION_PROMPT, SKILL_GENERATION_PROMPT
from backend.utils import extract_text_content
from backend.schemas import (
    AgentCreateRequest,
    AgentGenerateRequest,
    AgentSpec,
    SkillCreateRequest,
    SkillGenerateRequest,
    SkillSpec,
)

router = APIRouter(prefix="/api", tags=["agents", "skills"])


# ---- Agents ----

@router.get("/agents")
async def api_list_agents():
    agents = await asyncio.to_thread(list_agents)
    return [a.model_dump() for a in agents]


@router.get("/agents/{name}")
async def api_get_agent(name: str):
    a = await asyncio.to_thread(get_agent, name)
    if not a:
        return JSONResponse(status_code=404, content={"error": "Agent not found"})
    return a.model_dump()


@router.post("/agents")
async def api_create_agent(req: AgentCreateRequest):
    spec = AgentSpec.model_validate(req.model_dump())
    saved = await asyncio.to_thread(save_agent, spec)
    return saved.model_dump()


@router.put("/agents/{name}")
async def api_update_agent(name: str, req: AgentCreateRequest):
    existing = await asyncio.to_thread(get_agent, name)
    if not existing:
        return JSONResponse(status_code=404, content={"error": "Agent not found"})
    spec = existing.model_copy(update=req.model_dump())
    saved = await asyncio.to_thread(save_agent, spec)
    return saved.model_dump()


@router.delete("/agents/{name}")
async def api_delete_agent(name: str):
    if is_builtin_agent(name):
        return JSONResponse(
            status_code=409,
            content={
                "error": f"'{name}' is a built-in agent managed by the app and cannot be deleted.",
            },
        )
    return {"deleted": await asyncio.to_thread(delete_agent, name)}


@router.post("/agents/generate")
async def api_generate_agent(req: AgentGenerateRequest):
    """Use the configured LLM to generate an agent spec from a description."""
    cfg = await AppConfig.aload()
    cfg.apply_to_environ()

    from deep_agent.model_factory import create_llm
    llm = create_llm(cfg.llm.provider)

    from langchain_core.messages import HumanMessage, SystemMessage
    response = await llm.ainvoke([
        SystemMessage(content=AGENT_GENERATION_PROMPT),
        HumanMessage(content=req.user_description),
    ])

    content = extract_text_content(response.content)

    try:
        spec_data = json.loads(content)
        return spec_data
    except json.JSONDecodeError:
        return {"raw_response": content}


@router.get("/agents-config")
async def export_agents_json():
    """Export all agents as JSON."""
    agents = await asyncio.to_thread(list_agents)
    obj: dict[str, dict] = {}
    for a in agents:
        entry: dict[str, object] = {
            "description": a.description,
            "system_prompt": a.system_prompt,
        }
        if a.tools:
            entry["tools"] = a.tools
        if a.skills:
            entry["skills"] = a.skills
        if a.model_override:
            entry["model_override"] = a.model_override
        if a.subagent_llm_family:
            entry["subagent_llm_family"] = a.subagent_llm_family
        if a.mlx_model_id:
            entry["mlx_model_id"] = a.mlx_model_id
        obj[a.name] = entry
    return {"agents": obj}


@router.put("/agents-config")
async def save_agents_json(payload: dict):
    """Replace all agents from JSON.

    Accepts: ``{ "agents": { "name": { "description": "...", ... } } }``
    """
    agents_data = payload.get("agents") or {}
    if not isinstance(agents_data, dict):
        return JSONResponse(status_code=400, content={"error": "Expected 'agents' to be an object"})

    errors: list[str] = []
    new_specs: list[AgentSpec] = []

    for name, spec in agents_data.items():
        if not isinstance(spec, dict):
            errors.append(f"'{name}': value must be an object")
            continue
        if not spec.get("description"):
            errors.append(f"'{name}': 'description' is required")
            continue
        new_specs.append(AgentSpec(
            name=name,
            description=spec["description"],
            system_prompt=spec.get("system_prompt", ""),
            tools=spec.get("tools", []),
            skills=spec.get("skills", []),
            model_override=spec.get("model_override"),
            subagent_llm_family=spec.get("subagent_llm_family"),
            mlx_model_id=spec.get("mlx_model_id"),
        ))

    if errors:
        return JSONResponse(status_code=422, content={"error": "Validation failed", "details": errors})

    def _sync_agents() -> None:
        existing = {a.name for a in list_agents()}
        incoming = {a.name for a in new_specs}
        # Built-ins are never deleted via the bulk-JSON path: if the user omits
        # one from their JSON, we silently keep it instead of stripping it,
        # mirroring the protection on the per-agent DELETE endpoint.
        for removed in existing - incoming:
            if is_builtin_agent(removed):
                continue
            delete_agent(removed)
        for spec in new_specs:
            save_agent(spec)

    await asyncio.to_thread(_sync_agents)

    return {"status": "saved", "count": len(new_specs)}


# ---- Skills ----

@router.get("/skills")
async def api_list_skills():
    skills = await asyncio.to_thread(list_skills)
    return [s.model_dump() for s in skills]


@router.get("/skills/{name}")
async def api_get_skill(name: str):
    s = await asyncio.to_thread(get_skill, name)
    if not s:
        return JSONResponse(status_code=404, content={"error": "Skill not found"})
    return s.model_dump()


@router.post("/skills")
async def api_create_skill(req: SkillCreateRequest):
    spec = SkillSpec(
        name=req.name,
        description=req.description,
        content=req.content,
    )
    saved = await asyncio.to_thread(save_skill, spec)
    return saved.model_dump()


@router.put("/skills/{name}")
async def api_update_skill(name: str, req: SkillCreateRequest):
    spec = SkillSpec(
        name=req.name,
        description=req.description,
        content=req.content,
    )
    saved = await asyncio.to_thread(save_skill, spec)
    return saved.model_dump()


@router.delete("/skills/{name}")
async def api_delete_skill(name: str):
    if is_builtin_skill(name):
        return JSONResponse(
            status_code=409,
            content={
                "error": f"'{name}' is a built-in skill managed by the app and cannot be deleted.",
            },
        )
    agents = await asyncio.to_thread(list_agents)
    referencing_agents = [a.name for a in agents if name in a.skills]
    if referencing_agents:
        return JSONResponse(status_code=409, content={"error": f"Cannot delete: referenced by agent(s): {', '.join(referencing_agents)}"})
    return {"deleted": await asyncio.to_thread(delete_skill, name)}


@router.post("/skills/generate")
async def api_generate_skill(req: SkillGenerateRequest):
    cfg = await AppConfig.aload()
    cfg.apply_to_environ()

    from deep_agent.model_factory import create_llm
    llm = create_llm(cfg.llm.provider)

    from langchain_core.messages import HumanMessage, SystemMessage
    response = await llm.ainvoke([
        SystemMessage(content=SKILL_GENERATION_PROMPT),
        HumanMessage(content=req.user_description),
    ])

    content = extract_text_content(response.content)
    return {"content": content}


@router.get("/skills-config")
async def export_skills_json():
    """Export all skills as JSON."""
    skills = await asyncio.to_thread(list_skills)
    obj: dict[str, dict] = {}
    for s in skills:
        entry: dict[str, object] = {"description": s.description, "content": s.content}
        obj[s.name] = entry
    return {"skills": obj}


@router.put("/skills-config")
async def save_skills_json(payload: dict):
    """Replace all skills from JSON.

    Accepts: ``{ "skills": { "name": { "description": "...", "content": "..." } } }``
    """
    skills_data = payload.get("skills") or {}
    if not isinstance(skills_data, dict):
        return JSONResponse(status_code=400, content={"error": "Expected 'skills' to be an object"})

    errors: list[str] = []
    new_specs: list[SkillSpec] = []

    for name, spec in skills_data.items():
        if not isinstance(spec, dict):
            errors.append(f"'{name}': value must be an object")
            continue
        if not spec.get("content"):
            errors.append(f"'{name}': 'content' is required")
            continue
        new_specs.append(SkillSpec(
            name=name,
            description=spec.get("description", ""),
            content=spec["content"],
        ))

    if errors:
        return JSONResponse(status_code=422, content={"error": "Validation failed", "details": errors})

    def _sync_skills() -> None:
        existing = {s.name for s in list_skills()}
        incoming = {s.name for s in new_specs}
        # See _sync_agents: built-ins survive bulk-JSON edits even when the
        # user removes them from the payload.
        for removed in existing - incoming:
            if is_builtin_skill(removed):
                continue
            delete_skill(removed)
        for spec in new_specs:
            save_skill(spec)

    await asyncio.to_thread(_sync_skills)

    return {"status": "saved", "count": len(new_specs)}
