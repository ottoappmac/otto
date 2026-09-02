"""Example usage of Blender Agent within OTTO framework"""

import asyncio
from unittest.mock import MagicMock

# Import the new Blender agent
from agents.blender_agent_full import BlenderAgentFull

async def example_usage():
    """Demonstrate how Blender Agent works within OTTO"""
    
    # Mock LLM (in real usage, this would be an actual LLM)
    mock_llm = MagicMock()
    
    # Mock MCP client (Blender connection)
    mock_mcp = MagicMock()
    
    # Initialize the Blender Agent
    blender_agent = BlenderAgentFull(mock_llm, mock_mcp)
    
    print("=== Blender Agent Example ===")
    
    # Example 1: Character modeling request
    request1 = "Create a 3D character model for a fantasy game"
    print(f"\nUser Request: {request1}")
    
    result1 = await blender_agent.process_request(request1, "session_001")
    print("\nAgent Response:")
    print(result1)
    
    # Example 2: Architecture modeling request
    request2 = "Build a modern house with a garden"
    print(f"\nUser Request: {request2}")
    
    result2 = await blender_agent.process_request(request2, "session_002")
    print("\nAgent Response:")
    print(result2)
    
    # Example 3: Problem fixing request
    request3 = "My character model has topology issues"
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
    
    print("\n=== Agent Components ===")
    print(f"- Workflow Manager: {type(blender_agent.workflow_manager).__name__}")
    print(f"- Error Recovery: {type(blender_agent.error_recovery).__name__}")
    print(f"- Integration Tool: {type(blender_agent.blender_integration).__name__}")


if __name__ == "__main__":
    asyncio.run(example_usage())
