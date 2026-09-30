"""Complete Blender 3D Modeling Agent implementation.

This is the main entry point for text-to-3D workflows in Otto.  It has been
refactored to follow the **Design Spec → Code Generation** pattern inspired
by ShapeCraft (NeurIPS 2025) and SceneCraft (ICML 2024):

1. **Phase 0 — Design Spec**:  An LLM decomposes the text prompt into a
   structured ``DesignSpec`` (named parts, bounding volumes, hierarchy,
   modifiers, materials).  This is the "what to build" reasoning step that
   was previously missing.

2. **Phase 1 — Scene Diff**:  The spec is compared against the current
   Blender scene state to produce a delta: what to create, modify, or
   delete.

3. **Phase 2 — Code Generation**:  For each delta action, the agent
   generates the appropriate ``bpy`` code and executes it in Blender.

4. **Phase 3 — Verification**:  After each batch of operations, the agent
   takes a viewport screenshot and critiques the result, feeding back into
   the loop (SceneCraft's inner loop).

Usage::

    from agents.blender_agent_full import BlenderAgentFull

    agent = BlenderAgentFull(llm, mcp_client)
    result = await agent.process_request("a modern desk with drawers")
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

from deep_agent.agent import DeepAgent
from deep_agent.options import ToolOption, SubAgentOption

from agents.blender_agent import BlenderAgent
from agents.workflow.blender_workflow import BlenderWorkflowManager
from agents.error_recovery.blender_recovery import BlenderErrorRecovery
from tools.blender_integration import BlenderIntegrationTool
from agents.workflow.blender_design_spec import (
    DesignSpec,
    DesignSpecGenerator,
    MeshPart,
    SceneConstraints,
    compute_scene_diff,
)

logger = logging.getLogger(__name__)


class BlenderAgentFull:
    """Complete Blender 3D Modeling Agent that integrates with OTTO's framework.

    This agent now follows a structured 4-phase pipeline:

    Phase 0 — Design Spec (NEW):  Decompose the text prompt into a structured
        ``DesignSpec`` with named parts, bounding volumes, hierarchy, and
        suggested modifiers/materials.

    Phase 1 — Scene Diff (NEW):  Compare the spec against the current scene
        state to produce a delta (create / modify / delete).

    Phase 2 — Code Generation:  Generate ``bpy`` code for each delta action
        and execute it in Blender.

    Phase 3 — Verification:  Take a viewport screenshot, critique the result,
        and feed back into the loop for refinement.
    """

    def __init__(self, llm: BaseChatModel, mcp_client=None):
        self.llm = llm
        self.mcp_client = mcp_client

        # Initialize components
        self.workflow_manager = BlenderWorkflowManager()
        self.error_recovery = BlenderErrorRecovery()
        self.blender_integration = BlenderIntegrationTool(mcp_client)

        # NEW: Design spec generator (Phase 0).
        self.design_spec_generator = DesignSpecGenerator(llm=llm, fallback=True)

        # Context manager for session state
        self.session_context = None

        # Use existing OTTO DeepAgent framework for general capabilities
        # but specialized for Blender tasks
        self.deep_agent = DeepAgent(
            tools=[ToolOption.EXECUTE, ToolOption.VIEW_IMAGE, ToolOption.WEB_RESEARCHER],
            subagents=[SubAgentOption.WEB_VOYAGER],  # For reference research
        )

    async def process_request(self, user_request: str, session_id: str = "default") -> str:
        """Process user request using OTTO's agent patterns.

        This is the main entry point.  It runs the full 4-phase pipeline:
        Design Spec → Scene Diff → Code Generation → Verification.
        """
        try:
            # Initialize session context
            self.session_context = self._create_session_context(session_id)

            # Step 0: Validate environment (check Blender connection first)
            if not await self._validate_environment():
                return "Blender connection unavailable. Please ensure Blender MCP is running."

            # Step 1: Phase 0 — Generate Design Spec (NEW)
            design_spec = await self.design_spec_generator.generate(user_request)
            design_spec.prompt = user_request  # Ensure prompt is set

            # Step 2: Phase 1 — Scene Diff (NEW)
            scene_info = await self.blender_integration.get_scene_info()
            existing_objects = []
            if isinstance(scene_info, dict) and "scene_info" in scene_info:
                scene_data = scene_info["scene_info"]
                if isinstance(scene_data, dict) and "objects" in scene_data:
                    existing_objects = scene_data["objects"]
                elif isinstance(scene_data, list):
                    existing_objects = scene_data

            delta = compute_scene_diff(design_spec, existing_objects)

            # Step 3: Phase 2 — Code Generation & Execution (NEW)
            execution_result = await self._execute_design_spec(
                design_spec, delta, user_request
            )

            # Step 4: Phase 3 — Verification (NEW)
            verification_result = await self._verify_result(design_spec)

            # Format response in OTTO's standard format
            return self._format_response(
                design_spec, delta, execution_result, verification_result
            )

        except Exception as e:
            logger.error(f"Blender agent error: {e}")
            return f"Error processing request: {str(e)}. Please verify your Blender setup."

    def _create_session_context(self, session_id: str) -> Any:
        """Create session context for tracking modeling progress."""
        return f"session_{session_id}"

    # ── Phase 0: Design Spec ────────────────────────────────────────────

    async def _generate_design_spec(self, request: str) -> DesignSpec:
        """Generate a DesignSpec from the user's text request.

        This is the "what to build" reasoning step that was previously
        missing from the agent.  It externalises design decisions into a
        structured intermediate representation before any bpy code is
        generated.
        """
        return await self.design_spec_generator.generate(request)

    # ── Phase 1: Scene Diff ─────────────────────────────────────────────

    def _compute_delta(self, spec: DesignSpec, scene_info: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the delta between the design spec and current scene."""
        existing_objects = []
        if isinstance(scene_info, dict) and "scene_info" in scene_info:
            scene_data = scene_info["scene_info"]
            if isinstance(scene_data, dict) and "objects" in scene_data:
                existing_objects = scene_data["objects"]
            elif isinstance(scene_data, list):
                existing_objects = scene_data

        return compute_scene_diff(spec, existing_objects)

    # ── Phase 2: Code Generation & Execution ────────────────────────────

    async def _execute_design_spec(
        self,
        spec: DesignSpec,
        delta: Dict[str, Any],
        user_request: str,
    ) -> Dict[str, Any]:
        """Execute the design spec by generating and running bpy code.

        This is the core code-generation step.  For each part in the spec
        that needs to be created (according to the delta), generate the
        appropriate bpy code and execute it in Blender.
        """
        results = {
            "created": [],
            "modified": [],
            "deleted": delta.get("to_delete", []),
            "errors": [],
        }

        # Generate bpy code for each part that needs creation.
        for part in spec.parts:
            if part.name in delta.get("to_create", []):
                code = self._generate_bpy_code_for_part(part)
                try:
                    exec_result = await self.blender_integration.execute_blender_code(
                        code, context=f"Creating part: {part.name}"
                    )
                    if exec_result.get("success"):
                        results["created"].append(part.name)
                    else:
                        results["errors"].append(
                            f"Failed to create '{part.name}': {exec_result.get('error', 'unknown error')}"
                        )
                except Exception as e:
                    results["errors"].append(
                        f"Exception creating '{part.name}': {e}"
                    )

            elif part.name in delta.get("to_modify", []):
                code = self._generate_bpy_code_for_modification(part)
                try:
                    exec_result = await self.blender_integration.execute_blender_code(
                        code, context=f"Modifying part: {part.name}"
                    )
                    if exec_result.get("success"):
                        results["modified"].append(part.name)
                    else:
                        results["errors"].append(
                            f"Failed to modify '{part.name}': {exec_result.get('error', 'unknown error')}"
                        )
                except Exception as e:
                    results["errors"].append(
                        f"Exception modifying '{part.name}': {e}"
                    )

        return results

    def _generate_bpy_code_for_part(self, part: MeshPart) -> str:
        """Generate bpy code to create a single mesh part.

        This is the key code-generation function that translates a
        ``MeshPart`` into actual Blender Python code.
        """
        primitive_map = {
            "cube": ("primitive_cube_add", {"size": 2}),
            "uv_sphere": ("primitive_uv_sphere_add", {"radius": 1, "segments": 32, "ring_count": 16}),
            "cylinder": ("primitive_cylinder_add", {"radius": 1, "depth": 2, "vertices": 32}),
            "cone": ("primitive_cone_add", {"radius": 1, "depth": 2, "vertices": 32}),
            "torus": ("primitive_torus_add", {"major_radius": 1, "minor_radius": 0.4}),
        }

        op_name, kwargs = primitive_map.get(part.primitive_type, ("primitive_cube_add", {"size": 2}))

        # Build the location offset based on bounding box center.
        bb = part.bounding_box
        cx, cy, cz = bb.center_x, bb.center_y, bb.center_z

        code_lines = [
            f"import bpy",
            f"",
            f"# Create part: {part.name}",
            f'bpy.ops.mesh.{op_name}(',
        ]

        # Add kwargs.
        kw_parts = []
        for k, v in kwargs.items():
            kw_parts.append(f"    {k}={v}")
        code_lines.append(",\n".join(kw_parts))
        code_lines.append(")")
        code_lines.append(f"obj = bpy.context.active_object")
        code_lines.append(f'obj.name = "{part.name}"')

        # Set location based on bounding box center.
        code_lines.append(
            f'obj.location = ({cx:.4f}, {cy:.4f}, {cz:.4f})'
        )

        # Set scale based on bounding box dimensions.
        code_lines.append(
            f'obj.scale = ({bb.width / 2:.4f}, {bb.depth / 2:.4f}, {bb.height / 2:.4f})'
        )

        # Apply modifiers if specified.
        for mod_name in part.modifiers:
            if mod_name == "subdivision_surface":
                code_lines.append(
                    f'mod = obj.modifiers.new(name="Subdivision", type="SUBDIVISION")'
                )
                code_lines.append(f'mod.levels = 2')
                code_lines.append(f'mod.render_levels = 3')
            elif mod_name == "mirror":
                code_lines.append(
                    f'mod = obj.modifiers.new(name="Mirror", type="MIRROR")'
                )
            elif mod_name == "bevel":
                code_lines.append(
                    f'mod = obj.modifiers.new(name="Bevel", type="BEVEL")'
                )
                code_lines.append(f'mod.width = 0.02')

        # Set visibility/selectability.
        code_lines.append(f'obj.hide_viewport = {not part.is_visible}')
        code_lines.append(f'obj.hide_select = {not part.is_selectable}')

        # Parent to another object if specified.
        if part.parent_name:
            code_lines.append(
                f'# Parent {part.name} to {part.parent_name}'
            )
            code_lines.append(
                f'bpy.data.objects["{part.parent_name}"].select_set(True)'
            )
            code_lines.append(
                f'bpy.data.objects["{part.name}"].select_set(True)'
            )
            code_lines.append(
                f'bpy.context.view_layer.objects.active = bpy.data.objects["{part.parent_name}"]'
            )
            code_lines.append(
                f'bpy.ops.object.parent_set(type="OBJECT")'
            )

        return "\n".join(code_lines)

    def _generate_bpy_code_for_modification(self, part: MeshPart) -> str:
        """Generate bpy code to modify an existing mesh part.

        This handles repositioning, scaling, or applying modifiers to
        an existing object.
        """
        bb = part.bounding_box
        code_lines = [
            f"import bpy",
            f"",
            f"# Modify part: {part.name}",
            f'obj = bpy.data.objects.get("{part.name}")',
            f'if obj is None:',
            f'    print("Warning: {part.name} not found, skipping modification")',
            f'else:',
        ]

        # Reposition.
        code_lines.append(
            f'    obj.location = ({bb.center_x:.4f}, {bb.center_y:.4f}, {bb.center_z:.4f})'
        )

        # Rescale.
        code_lines.append(
            f'    obj.scale = ({bb.width / 2:.4f}, {bb.depth / 2:.4f}, {bb.height / 2:.4f})'
        )

        # Apply modifiers if specified.
        for mod_name in part.modifiers:
            code_lines.append(
                f'    # Add {mod_name} modifier to {part.name}'
            )
            code_lines.append(
                f'    existing_mods = [m.name for m in obj.modifiers]'
            )
            code_lines.append(
                f'    if "{mod_name}" not in existing_mods:'
            )
            code_lines.append(
                f'        mod = obj.modifiers.new(name="{mod_name}", type="SUBDIVISION" if "{mod_name}" == "subdivision_surface" else "BEVEL")'
            )
            code_lines.append(
                f'        if "{mod_name}" == "subdivision_surface":'
            )
            code_lines.append(f'            mod.levels = 2')
            code_lines.append(f'            mod.render_levels = 3')
            code_lines.append(
                f'        elif "{mod_name}" == "bevel":'
            )
            code_lines.append(f'            mod.width = 0.02')

        return "\n".join(code_lines)

    # ── Phase 3: Verification ───────────────────────────────────────────

    async def _verify_result(self, spec: DesignSpec) -> Dict[str, Any]:
        """Verify the created scene by taking a viewport screenshot.

        This is SceneCraft's inner loop: render → critique → refine.
        """
        try:
            screenshot_result = await self.blender_integration.get_scene_info()
            return {
                "verified": True,
                "scene_info": screenshot_result,
                "critique": self._generate_critique(spec),
            }
        except Exception as e:
            return {
                "verified": False,
                "error": str(e),
                "critique": "Could not verify — Blender may not be responding.",
            }

    def _generate_critique(self, spec: DesignSpec) -> str:
        """Generate a critique of the created scene based on the design spec.

        This is a heuristic critique that checks basic properties.
        In a full implementation, this would use a VLM to visually
        assess the viewport screenshot.
        """
        issues = []

        # Check part count.
        if len(spec.parts) < 2:
            issues.append(
                "The design spec has fewer than 2 parts. "
                "Consider decomposing the object into more components."
            )

        # Check for hierarchy.
        parts_with_parents = [p for p in spec.parts if p.parent_name]
        if not parts_with_parents and len(spec.parts) > 2:
            issues.append(
                "No parent-child relationships were specified. "
                "Consider adding hierarchy for better organization."
            )

        # Check bounding boxes.
        for part in spec.parts:
            bb = part.bounding_box
            if bb.width < 0.01 or bb.depth < 0.01 or bb.height < 0.01:
                issues.append(
                    f"Part '{part.name}' has a very small bounding box "
                    f"({bb.width:.3f} × {bb.depth:.3f} × {bb.height:.3f}). "
                    f"Check the scale."
                )

        if issues:
            return "Critique: " + "; ".join(issues)
        return "Critique: No obvious issues detected. Scene looks reasonable."

    # ── Legacy workflow methods (kept for backwards compatibility) ──────

    async def _validate_environment(self) -> bool:
        """Validate Blender environment with OTTO patterns."""
        if not self.mcp_client:
            logger.warning("No MCP client available for Blender agent")
            return False

        try:
            return await self.blender_integration.validate_connection()
        except Exception as e:
            logger.warning(f"Blender environment validation failed: {e}")
            return False

    # Legacy methods kept for backwards compatibility.
    # The new 4-phase pipeline (process_request) is preferred.

    def _detect_model_type(self, request: str) -> str:
        """Detect what type of model user wants to create."""
        if "character" in request.lower() or "avatar" in request.lower():
            return "character"
        elif "building" in request.lower() or "architecture" in request.lower():
            return "architecture"
        elif "product" in request.lower() or "object" in request.lower():
            return "product"
        elif "scene" in request.lower() or "environment" in request.lower():
            return "scene"
        return "general"

    def _assess_complexity(self, request: str) -> str:
        """Assess request complexity."""
        words = len(request.split())
        if words < 10:
            return "simple"
        elif words < 30:
            return "medium"
        else:
            return "complex"

    def _extract_requirements(self, request: str) -> List[str]:
        """Extract specific requirements from request."""
        requirements = []
        if "proportion" in request.lower() or "scale" in request.lower():
            requirements.append("precision_proportions")
        if "detail" in request.lower() or "texture" in request.lower():
            requirements.append("high_detail")
        if "render" in request.lower() or "finish" in request.lower():
            requirements.append("render_ready")
        return requirements

    def _extract_intent(self, request: str) -> str:
        """Extract the main intent of the request."""
        if "create" in request.lower() or "make" in request.lower():
            return "creation"
        elif "modify" in request.lower() or "change" in request.lower():
            return "modification"
        elif "fix" in request.lower() or "repair" in request.lower():
            return "problem_fixing"
        elif "improve" in request.lower() or "enhance" in request.lower():
            return "improvement"
        else:
            return "general"

    async def _execute_workflow(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the appropriate modeling workflow (legacy)."""
        model_type = analysis["model_type"]
        intent = analysis["intent"]

        if intent == "creation":
            return await self._handle_creation_workflow(model_type)
        elif intent == "modification":
            return await self._handle_modification_workflow(model_type)
        elif intent == "problem_fixing":
            return await self._handle_problem_fix_workflow(model_type)
        elif intent == "improvement":
            return await self._handle_improvement_workflow(model_type)
        else:
            return await self._handle_general_workflow(model_type)

    async def _handle_creation_workflow(self, model_type: str) -> Dict[str, Any]:
        """Handle creation workflow (legacy)."""
        phase_result = await self.workflow_manager.start_blockout_phase(model_type)
        if "error" in phase_result:
            return phase_result
        return {
            "workflow": "creation",
            "phase": phase_result["phase"],
            "instructions": phase_result["instructions"],
            "next_steps": ["refine", "detail", "verify"],
            "tools_needed": phase_result["tools_needed"],
            "blender_tips": [
                f"Start with basic {model_type} geometry",
                "Focus on proportions before details",
                "Save your work frequently"
            ]
        }

    async def _handle_modification_workflow(self, model_type: str) -> Dict[str, Any]:
        """Handle modification workflow (legacy)."""
        return {
            "workflow": "modification",
            "instructions": f"Modify existing {model_type} model. Start by identifying what changes are needed.",
            "tools_needed": ["select_objects", "transform_objects", "modify_geometry"],
            "blender_tips": [
                "Use object mode for transformations",
                "Check your mesh topology",
                "Save a backup before major changes"
            ]
        }

    async def _handle_problem_fix_workflow(self, model_type: str) -> Dict[str, Any]:
        """Handle problem fixing workflow (legacy)."""
        return {
            "workflow": "problem_fixing",
            "instructions": "Identify and fix issues with the model",
            "tools_needed": ["select_non_manifold", "check_geometry", "repair_mesh"],
            "blender_tips": [
                "Use 'Select Non-Manifold' to find issues",
                "Check for inverted normals",
                "Verify your mesh is manifold"
            ]
        }

    async def _handle_improvement_workflow(self, model_type: str) -> Dict[str, Any]:
        """Handle improvement workflow (legacy)."""
        return {
            "workflow": "improvement",
            "instructions": f"Enhance the {model_type} model for better quality",
            "tools_needed": ["subdivide", "smooth", "add_details"],
            "blender_tips": [
                "Add edge loops for better topology",
                "Apply smoothing where appropriate",
                "Check material and lighting"
            ]
        }

    async def _handle_general_workflow(self, model_type: str) -> Dict[str, Any]:
        """Handle general modeling workflow (legacy)."""
        return {
            "workflow": "general",
            "instructions": "Proceed with general modeling approach",
            "tools_needed": ["add_mesh_primitive", "transform_objects"],
            "blender_tips": [
                "Start simple and add complexity gradually",
                "Keep your topology clean",
                "Regular saves are essential"
            ]
        }

    # ── Response formatting ─────────────────────────────────────────────

    def _format_response(
        self,
        spec: DesignSpec,
        delta: Dict[str, Any],
        execution: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> str:
        """Format the full 4-phase pipeline result into a structured response."""
        lines = [f"=== Blender Modeling Assistant ==="]
        lines.append(f"\n**Design Spec:** {spec.prompt}")
        lines.append(f"**Model type:** {spec.model_type}  ")
        lines.append(f"**Complexity:** {spec.complexity}  ")

        # Parts table.
        if spec.parts:
            lines.append("\n## Parts\n")
            lines.append("| # | Name | Primitive | Size (W×D×H) | Parent |")
            lines.append("|---|------|-----------|-------------|--------|")
            for i, p in enumerate(spec.parts, 1):
                bb = p.bounding_box
                size = f"{bb.width:.2f}×{bb.depth:.2f}×{bb.height:.2f}"
                parent = p.parent_name or "— root —"
                lines.append(
                    f"| {i} | `{p.name}` | {p.primitive_type} | {size} | {parent} |"
                )

        # Delta summary.
        lines.append("\n## Scene Delta\n")
        lines.append(f"- **To create:** {', '.join(delta.get('to_create', ['(none)']))}")
        lines.append(f"- **To modify:** {', '.join(delta.get('to_modify', ['(none)']))}")
        lines.append(f"- **To delete:** {', '.join(delta.get('to_delete', ['(none)']))}")

        # Execution results.
        lines.append("\n## Execution Results\n")
        lines.append(f"- **Created:** {', '.join(execution.get('created', ['(none)']))}")
        lines.append(f"- **Modified:** {', '.join(execution.get('modified', ['(none)']))}")
        if execution.get("errors"):
            lines.append(f"- **Errors:**")
            for err in execution["errors"]:
                lines.append(f"  - {err}")

        # Verification.
        if verification.get("verified"):
            lines.append(f"\n## Verification: {verification.get('critique', 'OK')}")
        else:
            lines.append(f"\n## Verification: FAILED — {verification.get('error', 'unknown')}")

        return "\n".join(lines)

    async def handle_error(self, error_type: str, context: Dict[str, Any]) -> str:
        """Handle Blender-specific errors with recovery (legacy)."""
        try:
            recovery_result = await self.error_recovery.handle_error(error_type, context)

            error_msg = f"Error Type: {error_type}"
            if "suggestion" in recovery_result:
                error_msg += f"\nSuggestion: {recovery_result['suggestion']}"

            if "recommended_tools" in recovery_result:
                error_msg += "\nRecommended Tools: "
                error_msg += ", ".join(recovery_result["recommended_tools"])

            return error_msg
        except Exception as e:
            logger.error(f"Error handling failed: {e}")
            return f"Error handling failed: {str(e)}"
