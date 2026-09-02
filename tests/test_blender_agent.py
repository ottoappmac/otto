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
