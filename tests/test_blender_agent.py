"""Tests for Blender Agent implementation"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agents.blender_agent import BlenderAgent
from agents.blender_agent_full import BlenderAgentFull
from agents.workflow.blender_workflow import BlenderWorkflowManager
from agents.error_recovery.blender_recovery import BlenderErrorRecovery
from tools.blender_integration import BlenderIntegrationTool


class TestBlenderAgent:
    
    def test_blender_agent_initialization(self):
        """Test that BlenderAgent can be initialized"""
        # Mock LLM
        mock_llm = MagicMock()
        
        # Create agent
        agent = BlenderAgent(mock_llm)
        
        # Verify initialization
        assert agent.llm == mock_llm
        assert hasattr(agent, 'deep_agent')
        
    def test_blender_agent_analysis(self):
        """Test request analysis functionality"""
        mock_llm = MagicMock()
        agent = BlenderAgent(mock_llm)
        
        # Test character detection
        request = "Create a 3D character model"
        analysis = agent._analyze_request(request)
        assert analysis["model_type"] == "character"
        
        # Test architecture detection
        request = "Build a house for a client"
        analysis = agent._analyze_request(request)
        assert analysis["model_type"] == "architecture"
        
        # Test product detection
        request = "Model a product for e-commerce"
        analysis = agent._analyze_request(request)
        assert analysis["model_type"] == "product"
        
        # Test general detection
        request = "Make a 3D model"
        analysis = agent._analyze_request(request)
        assert analysis["model_type"] == "general"
        
    def test_model_type_detection(self):
        """Test specific model type detection"""
        mock_llm = MagicMock()
        agent = BlenderAgent(mock_llm)
        
        # Test character
        assert agent._detect_model_type("character model") == "character"
        assert agent._detect_model_type("avatar") == "character"
        
        # Test architecture
        assert agent._detect_model_type("building") == "architecture"
        assert agent._detect_model_type("architecture") == "architecture"
        
        # Test product
        assert agent._detect_model_type("product") == "product"
        assert agent._detect_model_type("object") == "product"
        
        # Test general
        assert agent._detect_model_type("random model") == "general"


class TestBlenderAgentFull:
    
    def test_blender_agent_full_initialization(self):
        """Test that BlenderAgentFull can be initialized"""
        # Mock LLM and MCP client
        mock_llm = MagicMock()
        mock_mcp = MagicMock()
        
        # Create agent
        agent = BlenderAgentFull(mock_llm, mock_mcp)
        
        # Verify initialization
        assert agent.llm == mock_llm
        assert agent.mcp_client == mock_mcp
        assert hasattr(agent, 'workflow_manager')
        assert hasattr(agent, 'error_recovery')
        assert hasattr(agent, 'blender_integration')
        


class TestBlenderWorkflowManager:
    
    def test_workflow_manager_initialization(self):
        """Test that workflow manager can be initialized"""
        manager = BlenderWorkflowManager()
        assert manager.current_state == "idle"
        
    def test_blockout_phase(self):
        """Test block-out phase creation"""
        manager = BlenderWorkflowManager()
        
        # Test with character model
        result = manager.start_blockout_phase("character")
        assert "phase" in result
        assert result["phase"] == "blockout"
        assert "instructions" in result
        


class TestBlenderErrorRecovery:
    
    def test_error_recovery_initialization(self):
        """Test that error recovery can be initialized"""
        recovery = BlenderErrorRecovery()
        assert hasattr(recovery, 'error_patterns')
        
    def test_error_handling(self):
        """Test error handling"""
        recovery = BlenderErrorRecovery()
        
        # Test non-manifold error
        result = recovery.handle_error("non_manifold_geometry", {})
        assert "suggestion" in result
        
        # Test unknown error
        result = recovery.handle_error("unknown_error", {})
        assert "suggestion" in result
        

class TestBlenderIntegration:
    
    def test_integration_initialization(self):
        """Test that Blender integration can be initialized"""
        # Test without client
        integration = BlenderIntegrationTool()
        assert integration.connection_status == "disconnected"
        
        # Test with mock client
        mock_client = MagicMock()
        integration = BlenderIntegrationTool(mock_client)
        assert integration.mcp_client == mock_client


# ── NEW TESTS: Design Spec module ──────────────────────────────────────

from agents.workflow.blender_design_spec import (
    DesignSpec,
    DesignSpecGenerator,
    MeshPart,
    SceneConstraints,
    BoundingBox,
    compute_scene_diff,
)


class TestBoundingBox:
    def test_properties(self):
        bb = BoundingBox(min_x=-1, min_y=-0.5, min_z=0, max_x=1, max_y=0.5, max_z=2)
        assert bb.width == 2.0
        assert bb.depth == 1.0
        assert bb.height == 2.0
        assert bb.center_x == 0.0
        assert bb.center_y == 0.0
        assert bb.center_z == 1.0


class TestMeshPart:
    def test_defaults(self):
        part = MeshPart(name="test_part")
        assert part.name == "test_part"
        assert part.primitive_type == "cube"
        assert part.parent_name is None
        assert part.modifiers == []
        assert part.materials == []
        assert part.is_visible is True


class TestDesignSpec:
    def test_to_markdown(self):
        spec = DesignSpec(
            prompt="test prompt",
            model_type="product",
            parts=[
                MeshPart(name="top", primitive_type="cube",
                         bounding_box=BoundingBox(min_x=-0.5, min_y=-0.25, min_z=0,
                                                  max_x=0.5, max_y=0.25, max_z=0.05)),
            ],
        )
        md = spec.to_markdown()
        assert "test prompt" in md
        assert "product" in md
        assert "top" in md

    def test_to_dict(self):
        spec = DesignSpec(prompt="test", model_type="general")
        d = spec.to_dict()
        assert d["prompt"] == "test"
        assert d["model_type"] == "general"


class TestDesignSpecGenerator:
    def test_heuristic_product(self):
        gen = DesignSpecGenerator(fallback=True)
        spec = gen._heuristic_parse("a modern office chair with wheels")
        assert spec.model_type == "product"
        names = [p.name for p in spec.parts]
        assert "seat" in names
        assert "back" in names
        assert "legs" in names

    def test_heuristic_character(self):
        gen = DesignSpecGenerator(fallback=True)
        spec = gen._heuristic_parse("a fantasy knight character with sword")
        assert spec.model_type == "character"
        names = [p.name for p in spec.parts]
        assert "body" in names
        assert "head" in names
        assert "left_arm" in names

    def test_heuristic_architecture(self):
        gen = DesignSpecGenerator(fallback=True)
        spec = gen._heuristic_parse("a modern house with large windows")
        assert spec.model_type == "architecture"
        names = [p.name for p in spec.parts]
        assert "windows" in names

    def test_complexity_simple(self):
        gen = DesignSpecGenerator(fallback=True)
        spec = gen._heuristic_parse("chair")
        assert spec.complexity == "simple"

    def test_complexity_complex(self):
        gen = DesignSpecGenerator(fallback=True)
        long_prompt = " ".join(["a", "very", "detailed", "modern", "office",
                                "chair", "with", "adjustable", "height",
                                "lumbar", "support", "armrests", "and",
                                "breathable", "mesh", "backrest", "with",
                                "five", "wheels", "and", "chrome",
                                "base", "for", "a", "premium",
                                "ergonomic", "design"])
        spec = gen._heuristic_parse(long_prompt)
        assert spec.complexity == "complex"


class TestComputeSceneDiff:
    def test_create_only(self):
        spec = DesignSpec(prompt="test", parts=[MeshPart(name="new_part")])
        diff = compute_scene_diff(spec, [{"name": "old_thing"}])
        assert diff["to_create"] == ["new_part"]
        assert diff["to_delete"] == ["old_thing"]
        assert diff["to_modify"] == []

    def test_modify_only(self):
        spec = DesignSpec(prompt="test", parts=[MeshPart(name="existing")])
        diff = compute_scene_diff(spec, [{"name": "existing"}])
        assert diff["to_modify"] == ["existing"]
        assert diff["to_create"] == []
        assert diff["to_delete"] == []

    def test_empty_scene(self):
        spec = DesignSpec(prompt="test", parts=[MeshPart(name="part1")])
        diff = compute_scene_diff(spec, [])
        assert diff["to_create"] == ["part1"]
        assert diff["to_delete"] == []
