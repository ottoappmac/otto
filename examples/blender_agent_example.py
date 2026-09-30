"""Example usage of Blender Agent within OTTO framework.

Demonstrates the new 4-phase pipeline:
  Phase 0 — Design Spec (NEW):  Decompose text prompt into structured spec.
  Phase 1 — Scene Diff (NEW):   Compare spec against current scene state.
  Phase 2 — Code Generation:    Generate and execute bpy code for each part.
  Phase 3 — Verification (NEW):  Critique the result and feed back.
"""

import asyncio
from unittest.mock import MagicMock

# Import the new Blender agent
from agents.blender_agent_full import BlenderAgentFull


async def example_usage():
    """Demonstrate how Blender Agent works within OTTO."""

    # Mock LLM (in real usage, this would be an actual LLM)
    mock_llm = MagicMock()

    # Mock MCP client (Blender connection)
    mock_mcp = MagicMock()

    # Initialize the Blender Agent
    blender_agent = BlenderAgentFull(mock_llm, mock_mcp)

    print("=== Blender Agent Example (4-Phase Pipeline) ===")

    # Example 1: Product modeling (chair)
    request1 = "Create a modern office chair with wheels"
    print(f"\nUser Request: {request1}")

    result1 = await blender_agent.process_request(request1, "session_001")
    print("\nAgent Response:")
    print(result1)

    # Example 2: Architecture modeling (house)
    request2 = "Build a modern house with large windows and a garden"
    print(f"\nUser Request: {request2}")

    result2 = await blender_agent.process_request(request2, "session_002")
    print("\nAgent Response:")
    print(result2)

    # Example 3: Character modeling
    request3 = "Create a fantasy knight character with sword and shield"
    print(f"\nUser Request: {request3}")

    result3 = await blender_agent.process_request(request3, "session_003")
    print("\nAgent Response:")
    print(result3)

    # Example 4: Error handling
    print("\n=== Error Handling Example ===")
    error_result = await blender_agent.handle_error(
        "non_manifold_geometry",
        {"context": "user_model"}
    )
    print("Error Response:")
    print(error_result)

    # Example 5: Design spec (standalone)
    print("\n=== Design Spec Example (standalone) ===")
    from agents.workflow.blender_design_spec import DesignSpecGenerator

    gen = DesignSpecGenerator(fallback=True)
    spec = gen._heuristic_parse("a modern desk with drawers")
    print(spec.to_markdown())

    print("\n=== Agent Components ===")
    print(f"- Workflow Manager: {type(blender_agent.workflow_manager).__name__}")
    print(f"- Error Recovery: {type(blender_agent.error_recovery).__name__}")
    print(f"- Integration Tool: {type(blender_agent.blender_integration).__name__}")
    print(f"- Design Spec Generator: {type(blender_agent.design_spec_generator).__name__}")


if __name__ == "__main__":
    asyncio.run(example_usage())
