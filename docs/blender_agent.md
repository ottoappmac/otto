# Blender Agent Documentation

## Overview

The Blender Agent is a specialized agent designed to assist users with 3D modeling tasks in Blender. It leverages the OTTO agent framework while providing tailored guidance for Blender's specific workflows and challenges.

## Key Features

### 1. Specialized 3D Modeling Workflows
- **Iterative Modeling**: Supports the block-out → verify → detail → verify workflow
- **Phase-based Guidance**: Provides step-by-step instructions for each modeling phase
- **Model Type Detection**: Automatically identifies character, architecture, product, and scene modeling tasks

### 2. Error Recovery & Prevention
- **Error Detection**: Identifies common Blender issues like non-manifold geometry
- **Recovery Strategies**: Provides actionable solutions for common modeling problems
- **Connection Management**: Handles Blender MCP connection states gracefully

### 3. Context Awareness
- **Session Tracking**: Maintains modeling progress and state across interactions
- **Tool Integration**: Works seamlessly with Blender's toolset through MCP connections
- **Workflow Management**: Guides users through optimized modeling processes

## Integration with OTTO

The Blender Agent integrates with the existing OTTO agent framework by:

- Using `DeepAgent` as the base orchestrator
- Leveraging existing tools like `execute` and `view_image`
- Following OTTO's error handling and logging patterns
- Maintaining compatibility with the agent library system

## Usage Examples

### Creating a Character Model
```
User: "Create a 3D character model for a fantasy game"
Blender Agent: "Starting with character modeling workflow... Begin with a UV sphere for the body shape. Focus on proportions first before adding details."
```

### Architecture Modeling
```
User: "Build a house with a modern design"
Blender Agent: "Architecture modeling workflow initiated. Start with a basic cube structure. Pay attention to scale references and measurement tools."
```

### Problem Fixing
```
User: "My model has topology issues"
Blender Agent: "I've detected topology issues. Use 'Select Non-Manifold' (Shift+Alt+Ctrl+M) to identify problem areas. Try reorganizing edge loops to maintain clean topology."
```

## Components

### 1. BlenderAgent (src/agents/blender_agent.py)
- Core agent class that handles request processing
- Request analysis and model type detection
- Basic workflow orchestration

### 2. BlenderAgentFull (src/agents/blender_agent_full.py)
- Complete implementation with full workflow integration
- Advanced session management
- Error recovery system integration

### 3. BlenderWorkflowManager (src/agents/workflow/blender_workflow.py)
- Manages modeling workflows with phase tracking
- Implements iterative modeling processes
- Provides step-by-step guidance

### 4. BlenderErrorRecovery (src/agents/error_recovery/blender_recovery.py)
- Handles Blender-specific errors and issues
- Provides recovery strategies for common problems
- Integrates with OTTO's error handling patterns

### 5. BlenderIntegrationTool (src/tools/blender_integration.py)
- Specialized tool for Blender MCP integration
- Connection validation and management
- Safe execution with timeout handling

## Best Practices for Users

1. **Environment Setup**: Ensure Blender MCP is running and connected
2. **Regular Saves**: Always save your work during modeling sessions
3. **Progress Tracking**: Use the agent's guidance to stay on track
4. **Error Response**: Follow the agent's suggestions when issues arise
5. **Workflow Adherence**: Follow the phased modeling approach for best results

## Configuration

The Blender Agent works with existing OTTO configuration patterns:

```env
# Blender MCP settings (if using external connection)
BLENDER_MCP_HOST=localhost
BLENDER_MCP_PORT=8932
```

## Testing

Unit tests are provided in `tests/test_blender_agent.py` that verify:

- Agent initialization and core functionality
- Request analysis and model type detection
- Workflow management capabilities
- Error recovery systems
- Tool integration patterns
