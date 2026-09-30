"""Design-spec generation for text-to-3D workflows.

Inspired by ShapeCraft (NeurIPS 2025) and SceneCraft (ICML 2024), this module
externalises the "what to build" reasoning into a structured intermediate
representation before any ``bpy`` code is generated.

Key ideas:
  * **Component decomposition** — break a text prompt into named parts with
    spatial relationships (ShapeCraft's GPS nodes).
  * **Bounding-volume estimation** — estimate scale/proportions for each part
    (SceneCraft's layout optimisation).
  * **Constraint diff** — compare the spec against the current scene state and
    emit only the delta (what to add, move, or delete).

Usage::

    from agents.workflow.blender_design_spec import DesignSpecGenerator

    spec = DesignSpecGenerator(llm).generate("a modern desk with drawers")
    print(spec.to_markdown())
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data model ──────────────────────────────────────────────────────────────


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in Blender units (metres)."""
    min_x: float = -1.0
    min_y: float = -1.0
    min_z: float = 0.0
    max_x: float = 1.0
    max_y: float = 1.0
    max_z: float = 1.0

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        return self.max_y - self.min_y

    @property
    def height(self) -> float:
        return self.max_z - self.min_z

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) / 2.0

    @property
    def center_y(self) -> float:
        return (self.min_y + self.max_y) / 2.0

    @property
    def center_z(self) -> float:
        return (self.min_z + self.max_z) / 2.0


@dataclass
class MeshPart:
    """A single mesh component to be created in Blender.

    Mirrors ShapeCraft's graph-node concept: each part has a name, a
    suggested primitive type, a bounding volume, and a parent-child
    relationship.
    """
    name: str
    description: str = ""
    primitive_type: str = "cube"          # cube | sphere | cylinder | uv_sphere | cone | torus
    bounding_box: BoundingBox = field(default_factory=BoundingBox)
    parent_name: Optional[str] = None      # None → root object
    modifiers: List[str] = field(default_factory=list)
    # e.g. ["subdivision_surface", "bevel", "mirror", "array"]
    materials: List[str] = field(default_factory=list)
    # e.g. ["metal", "wood", "glass"]
    is_visible: bool = True
    is_selectable: bool = True


@dataclass
class SceneConstraints:
    """Global scene-level constraints."""
    scene_dimensions: Optional[BoundingBox] = None  # None = auto-fit
    ground_plane: bool = True
    default_lighting: str = "three_point"  # three_point | studio | outdoor | none
    camera_position: Optional[str] = None   # "front" | "iso" | "top" | None
    unit_system: str = "metric"            # metric | imperial


@dataclass
class DesignSpec:
    """Full design specification for a text-to-3D request.

    This is the "Phase 0" output that the agent should produce before
    generating any ``bpy`` code.
    """
    prompt: str
    model_type: str = "general"           # character | architecture | product | scene | prop
    complexity: str = "medium"            # simple | medium | complex
    parts: List[MeshPart] = field(default_factory=list)
    constraints: SceneConstraints = field(default_factory=SceneConstraints)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Render the spec as a human-readable markdown document."""
        lines = [f"# Design Spec: {self.prompt}"]
        lines.append(f"\n**Model type:** {self.model_type}  ")
        lines.append(f"**Complexity:** {self.complexity}  ")

        if self.parts:
            lines.append("\n## Parts\n")
            lines.append("| # | Name | Primitive | Size (W×D×H) | Parent |")
            lines.append("|---|------|-----------|-------------|--------|")
            for i, p in enumerate(self.parts, 1):
                bb = p.bounding_box
                size = f"{bb.width:.2f}×{bb.depth:.2f}×{bb.height:.2f}"
                parent = p.parent_name or "— root —"
                lines.append(
                    f"| {i} | `{p.name}` | {p.primitive_type} | {size} | {parent} |"
                )

        if self.constraints.scene_dimensions:
            sd = self.constraints.scene_dimensions
            lines.append(
                f"\n## Scene bounds: {sd.width:.2f} × {sd.depth:.2f} × {sd.height:.2f}"
            )

        if self.constraints.default_lighting != "three_point":
            lines.append(f"\n**Lighting:** {self.constraints.default_lighting}")

        if self.notes:
            lines.append("\n## Notes\n")
            for n in self.notes:
                lines.append(f"- {n}")

        return "\n".join(lines)


# ── Generator interface ─────────────────────────────────────────────────────


class DesignSpecGenerator:
    """Generate a ``DesignSpec`` from a text prompt using an LLM.

    The LLM is expected to return a JSON object matching the ``DesignSpec``
    schema.  If no LLM is provided, a heuristic fallback parser is used.

    Args:
        llm:      Optional LangChain ``BaseChatModel``.  When present, the
                  generator calls the LLM with a structured prompt.
        fallback: When ``llm`` is None, use heuristic keyword matching.
    """

    # Heuristic primitive-type keywords (fallback only).
    _PRIMITIVE_HINTS: Dict[str, List[str]] = {
        "uv_sphere": ["sphere", "ball", "globe", "head", "eye", "wheel"],
        "cylinder": ["cylinder", "column", "pipe", "tube", "leg", "arm", "trunk"],
        "cone": ["cone", "roof", "horn", "mountain", "spire"],
        "torus": ["torus", "donut", "ring", "tire", "bracelet"],
        "cube": ["box", "block", "table", "desk", "chair", "drawer", "wall",
                 "floor", "base", "plate", "frame", "body", "torso"],
    }

    # Heuristic size hints (in metres, approximate).
    _SIZE_HINTS: Dict[str, Dict[str, float]] = {
        "character": {"width": 0.5, "depth": 0.3, "height": 1.7},
        "architecture": {"width": 10.0, "depth": 8.0, "height": 3.0},
        "product": {"width": 0.5, "depth": 0.3, "height": 0.3},
        "prop": {"width": 0.3, "depth": 0.2, "height": 0.2},
        "scene": {"width": 20.0, "depth": 20.0, "height": 10.0},
    }

    def __init__(self, llm: Optional[Any] = None, *, fallback: bool = True):
        self._llm = llm
        self._fallback = fallback

    # ── Public API ──────────────────────────────────────────────────────

    async def generate(self, prompt: str) -> DesignSpec:
        """Generate a ``DesignSpec`` from *prompt*.

        If an LLM is available, call it with a structured prompt.
        Otherwise fall back to heuristic keyword matching.
        """
        if self._llm is not None:
            return await self._generate_with_llm(prompt)
        if self._fallback:
            return self._heuristic_parse(prompt)
        raise RuntimeError(
            "No LLM provided and fallback is disabled. "
            "Pass fallback=True or provide an llm= argument."
        )

    # ── LLM path ────────────────────────────────────────────────────────

    async def _generate_with_llm(self, prompt: str) -> DesignSpec:
        """Call the LLM with a structured prompt and parse the result."""
        system_prompt = (
            "You are a 3D design assistant.  Given a text prompt, produce a "
            "structured design specification as a JSON object with these keys:\n"
            "\n"
            "```\n"
            "{\n"
            '  "model_type": "character|architecture|product|scene|prop",\n'
            '  "complexity": "simple|medium|complex",\n'
            '  "parts": [\n'
            '    {\n'
            '      "name": "part_name",\n'
            '      "description": "what this part is",\n'
            '      "primitive_type": "cube|sphere|cylinder|uv_sphere|cone|torus",\n'
            '      "bounding_box": {"min_x": -1, "min_y": -1, "min_z": 0,\n'
            '                         "max_x": 1, "max_y": 1, "max_z": 1},\n'
            '      "parent_name": "optional parent part name or null",\n'
            '      "modifiers": ["subdivision_surface", "bevel"],\n'
            '      "materials": ["metal", "wood"]\n'
            '    }\n'
            "  ],\n"
            '  "constraints": {\n'
            '    "ground_plane": true,\n'
            '    "default_lighting": "three_point",\n'
            '    "camera_position": "front|iso|top|null"\n'
            "  },\n"
            '  "notes": ["optional notes"]\n'
            "}\n"
            "```\n"
            "\n"
            "Guidelines:\n"
            "- Decompose the object into 3-12 named parts.\n"
            "- Each part should have a reasonable bounding box (in metres).\n"
            "- Use parent_name to express hierarchy (null = root).\n"
            "- Suggest modifiers (subdivision_surface, bevel, mirror) where\n"
            "  appropriate.\n"
            "- Suggest materials (metal, wood, glass, fabric, plastic).\n"
            "- Camera position: 'front' for product shots, 'iso' for\n"
            "  architectural, 'top' for floor plans.\n"
        )

        try:
            response = await self._llm.ainvoke([
                ("system", system_prompt),
                ("human", prompt),
            ])
            text = response.content if hasattr(response, "content") else str(response)
            return self._parse_llm_json(text)
        except Exception as e:
            logger.warning("LLM design spec generation failed (%s), falling back", e)
            if self._fallback:
                return self._heuristic_parse(prompt)
            raise

    @staticmethod
    def _parse_llm_json(text: str) -> DesignSpec:
        """Extract a JSON object from LLM output (may be wrapped in markdown)."""
        # Strip markdown code fences if present.
        cleaned = text.strip()
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1)

        data = json.loads(cleaned)
        return DesignSpec(
            prompt="",  # will be set by caller
            model_type=data.get("model_type", "general"),
            complexity=data.get("complexity", "medium"),
            parts=[
                MeshPart(
                    name=p.get("name", f"part_{i}"),
                    description=p.get("description", ""),
                    primitive_type=p.get("primitive_type", "cube"),
                    bounding_box=BoundingBox(**p.get("bounding_box", {})),
                    parent_name=p.get("parent_name"),
                    modifiers=p.get("modifiers", []),
                    materials=p.get("materials", []),
                )
                for i, p in enumerate(data.get("parts", []))
            ],
            constraints=SceneConstraints(
                ground_plane=data.get("constraints", {}).get("ground_plane", True),
                default_lighting=data.get("constraints", {}).get(
                    "default_lighting", "three_point"
                ),
                camera_position=data.get("constraints", {}).get("camera_position"),
            ),
            notes=data.get("notes", []),
        )

    # ── Heuristic fallback ──────────────────────────────────────────────

    def _heuristic_parse(self, prompt: str) -> DesignSpec:
        """Parse a text prompt into a DesignSpec using keyword heuristics.

        This is a lightweight fallback when no LLM is available.  It
        recognises common object types and estimates bounding volumes.
        """
        lower = prompt.lower()

        # Detect model type.
        model_type = "general"
        for mt, keywords in [
            ("character", ["character", "avatar", "person", "human", "robot"]),
            ("architecture", ["building", "house", "room",
                              "architecture", "interior", "exterior"]),
            ("product", ["product", "furniture", "chair", "table", "desk",
                         "lamp", "vase", "phone", "laptop"]),
            ("prop", ["prop", "weapon", "tool", "accessory"]),
            ("scene", ["scene", "environment", "landscape", "garden"]),
        ]:
            if any(kw in lower for kw in keywords):
                model_type = mt
                break

        # Detect complexity.
        complexity = "medium"
        if len(prompt.split()) < 8:
            complexity = "simple"
        elif len(prompt.split()) > 30:
            complexity = "complex"

        # Heuristic part extraction.
        parts = self._extract_parts_heuristic(prompt, lower, model_type)

        # Size hints.
        size = self._SIZE_HINTS.get(model_type, {"width": 1.0, "depth": 1.0, "height": 1.0})

        # Build bounding boxes for root parts.
        for part in parts:
            bb = part.bounding_box
            half_w, half_d, half_h = size["width"] / 2, size["depth"] / 2, size["height"] / 2
            bb.min_x, bb.max_x = -half_w, half_w
            bb.min_y, bb.max_y = -half_d, half_d
            bb.min_z, bb.max_z = 0.0, half_h * 2

        return DesignSpec(
            prompt=prompt,
            model_type=model_type,
            complexity=complexity,
            parts=parts,
        )

    def _extract_parts_heuristic(
        self, prompt: str, lower: str, model_type: str
    ) -> List[MeshPart]:
        """Extract named parts from the prompt using keyword matching."""
        parts: List[MeshPart] = []

        # Character parts.
        if model_type == "character":
            parts.append(MeshPart(
                name="body", primitive_type="cylinder",
                description="Main torso and body",
            ))
            parts.append(MeshPart(
                name="head", primitive_type="uv_sphere",
                description="Head / face",
            ))
            parts.append(MeshPart(
                name="left_arm", primitive_type="cylinder",
                description="Left arm",
            ))
            parts.append(MeshPart(
                name="right_arm", primitive_type="cylinder",
                description="Right arm",
            ))
            parts.append(MeshPart(
                name="left_leg", primitive_type="cylinder",
                description="Left leg",
            ))
            parts.append(MeshPart(
                name="right_leg", primitive_type="cylinder",
                description="Right leg",
            ))
            for p in parts[1:]:  # head and limbs
                p.parent_name = "body"

        # Architecture parts.
        elif model_type == "architecture":
            if "wall" in lower or "room" in lower:
                parts.append(MeshPart(
                    name="walls", primitive_type="cube",
                    description="Room walls",
                ))
            if "floor" in lower:
                parts.append(MeshPart(
                    name="floor", primitive_type="cube",
                    description="Floor plane",
                ))
            if "window" in lower:
                parts.append(MeshPart(
                    name="windows", primitive_type="cube",
                    description="Window openings",
                ))
            if "door" in lower:
                parts.append(MeshPart(
                    name="doors", primitive_type="cube",
                    description="Door openings",
                ))
            if "roof" in lower or "gable" in lower:
                parts.append(MeshPart(
                    name="roof", primitive_type="cone",
                    description="Roof structure",
                ))
            if not parts:
                parts.append(MeshPart(
                    name="building", primitive_type="cube",
                    description="Main building volume",
                ))

        # Product / furniture parts.
        elif model_type == "product":
            if "chair" in lower:
                parts.append(MeshPart(name="seat", primitive_type="cube",
                                      description="Seat surface"))
                parts.append(MeshPart(name="back", primitive_type="cube",
                                      description="Back rest"))
                parts.append(MeshPart(name="legs", primitive_type="cylinder",
                                      description="Chair legs"))
            elif "table" in lower or "desk" in lower:
                parts.append(MeshPart(name="top", primitive_type="cube",
                                      description="Table/desk surface"))
                parts.append(MeshPart(name="legs", primitive_type="cylinder",
                                      description="Support legs"))
            elif "lamp" in lower:
                parts.append(MeshPart(name="base", primitive_type="cylinder",
                                      description="Lamp base"))
                parts.append(MeshPart(name="pole", primitive_type="cylinder",
                                      description="Lamp pole"))
                parts.append(MeshPart(name="shade", primitive_type="cone",
                                      description="Lamp shade"))
            elif "vase" in lower:
                parts.append(MeshPart(name="body", primitive_type="cylinder",
                                      description="Vase body"))
                parts.append(MeshPart(name="neck", primitive_type="cylinder",
                                      description="Vase neck"))
            else:
                parts.append(MeshPart(
                    name="object", primitive_type="cube",
                    description="Main product volume",
                ))

        # Prop parts.
        elif model_type == "prop":
            parts.append(MeshPart(
                name="main", primitive_type="cube",
                description="Main prop body",
            ))

        # Scene / environment.
        elif model_type == "scene":
            if "ground" in lower or "floor" in lower:
                parts.append(MeshPart(name="ground", primitive_type="cube",
                                      description="Ground plane"))
            if "tree" in lower:
                parts.append(MeshPart(name="trunks", primitive_type="cylinder",
                                      description="Tree trunks"))
                parts.append(MeshPart(name="foliage", primitive_type="uv_sphere",
                                      description="Tree foliage"))
            if "water" in lower or "pool" in lower:
                parts.append(MeshPart(name="water", primitive_type="cube",
                                      description="Water surface"))

        # Set parent relationships for hierarchy.
        root_names = [p.name for p in parts if p.parent_name is None]
        for part in parts:
            if part.parent_name is None and len(root_names) > 1:
                # Only one root; others are children of the first root.
                if part.name != root_names[0]:
                    part.parent_name = root_names[0]

        return parts


# ── Scene diff utility ──────────────────────────────────────────────────────


def compute_scene_diff(
    spec: DesignSpec,
    existing_objects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare a DesignSpec against existing scene objects.

    Returns a dict with keys ``to_create``, ``to_modify``, ``to_delete``
    containing lists of part names that need each action.

    Args:
        spec: The target design specification.
        existing_objects: List of dicts from ``get_scene_info()`` with at
            least a ``name`` key for each object.
    """
    existing_names = {obj.get("name", "") for obj in existing_objects}
    spec_names = {p.name for p in spec.parts}

    to_create = list(spec_names - existing_names)
    to_delete = list(existing_names - spec_names)
    to_modify = list(spec_names & existing_names)

    return {
        "to_create": to_create,
        "to_modify": to_modify,
        "to_delete": to_delete,
    }
