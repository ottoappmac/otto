"""Complete Blender 3D Modeling Agent implementation"""

from __future__ import annotations
import logging
from typing import Any, Optional, Dict, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from deep_agent.agent import DeepAgent
from deep_agent.options import ToolOption, SubAgentOption

from agents.blender_agent import BlenderAgent
from agents.workflow.blender_workflow import BlenderWorkflowManager
from agents.error_recovery.blender_recovery import BlenderErrorRecovery
from tools.blender_integration import BlenderIntegrationTool

logger = logging.getLogger(__name__)


class BlenderAgentFull:
    """Complete Blender 3D Modeling Agent that integrates with OTTO's framework"""
    
    def __init__(self, llm: BaseChatModel, mcp_client=None):
        self.llm = llm
        self.mcp_client = mcp_client
        
        # Initialize components
        self.workflow_manager = BlenderWorkflowManager()
        self.error_recovery = BlenderErrorRecovery()
        self.blender_integration = BlenderIntegrationTool(mcp_client)
        
        # Context manager for session state
        self.session_context = None
        
        # Use existing OTTO DeepAgent framework for general capabilities
        # but specialized for Blender tasks
        self.deep_agent = DeepAgent(
            tools=[ToolOption.EXECUTE, ToolOption.VIEW_IMAGE, ToolOption.WEB_RESEARCHER],
            subagents=[SubAgentOption.WEB_VOYAGER],  # For reference research
        )
        
    async def process_request(self, user_request: str, session_id: str = "default") -> str:
        """Process user request using OTTO's agent patterns"""
        try:
            # Initialize session context
            self.session_context = self._create_session_context(session_id)
            
            # Step 1: Analyze request
            analysis = await self._analyze_request(user_request)
            
            # Step 2: Validate environment
            if not await self._validate_environment():
                return "Blender connection unavailable. Please ensure Blender MCP is running."
                
            # Step 3: Select appropriate workflow
            workflow_result = await self._execute_workflow(analysis)
            
            # Step 4: Format response in OTTO's standard format
            return self._format_response(workflow_result)
            
        except Exception as e:
            logger.error(f"Blender agent error: {e}")
            return f"Error processing request: {str(e)}. Please verify your Blender setup."
            
    def _create_session_context(self, session_id: str) -> Any:
        """Create session context for tracking modeling progress"""
        # In a real implementation, this would use the existing context system
        return f"session_{session_id}"
        
    async def _analyze_request(self, request: str) -> Dict[str, Any]:
        """Analyze request following OTTO's pattern matching"""
        return {
            "model_type": self._detect_model_type(request),
            "complexity": self._assess_complexity(request),
            "requirements": self._extract_requirements(request),
            "intent": self._extract_intent(request)
        }
        
    def _detect_model_type(self, request: str) -> str:
        """Detect what type of model user wants to create"""
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
        """Assess request complexity"""
        words = len(request.split())
        if words < 10:
            return "simple"
        elif words < 30:
            return "medium"
        else:
            return "complex"
            
    def _extract_requirements(self, request: str) -> List[str]:
        """Extract specific requirements from request"""
        requirements = []
        # Extract common requirements
        if "proportion" in request.lower() or "scale" in request.lower():
            requirements.append("precision_proportions")
        if "detail" in request.lower() or "texture" in request.lower():
            requirements.append("high_detail")
        if "render" in request.lower() or "finish" in request.lower():
            requirements.append("render_ready")
        return requirements
        
    def _extract_intent(self, request: str) -> str:
        """Extract the main intent of the request"""
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
            
    async def _validate_environment(self) -> bool:
        """Validate Blender environment with OTTO patterns"""
        # Check if we have a valid MCP client
        if not self.mcp_client:
            logger.warning("No MCP client available for Blender agent")
            return False
            
        # Test connection
        try:
            # Use the existing integration tool to validate
            return await self.blender_integration.validate_connection()
        except Exception as e:
            logger.warning(f"Blender environment validation failed: {e}")
            return False
            
    async def _execute_workflow(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the appropriate modeling workflow"""
        model_type = analysis["model_type"]
        intent = analysis["intent"]
        
        # Execute appropriate workflow based on intent
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
        """Handle creation workflow"""
        # Start with appropriate workflow
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
        """Handle modification workflow"""
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
        """Handle problem fixing workflow"""
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
        """Handle improvement workflow"""
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
        """Handle general modeling workflow"""
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
        
    def _format_response(self, result: Dict[str, Any]) -> str:
        """Format response in OTTO's standard format"""
        # Format into a clear, structured response
        output_parts = []
        
        # Main workflow info
        output_parts.append("=== Blender Modeling Assistant ===")
        
        if "workflow" in result:
            output_parts.append(f"Workflow Type: {result['workflow'].title()}")
            
        if "phase" in result:
            output_parts.append(f"Current Phase: {result['phase'].title()}")
            
        # Instructions
        if "instructions" in result:
            output_parts.append(f"\nInstructions:")
            output_parts.append(f"{result['instructions']}")
            
        # Tools needed
        if "tools_needed" in result:
            output_parts.append(f"\nRecommended Tools:")
            for tool in result["tools_needed"]:
                output_parts.append(f"- {tool}")
                
        # Tips
        if "blender_tips" in result:
            output_parts.append(f"\nBlender Tips:")
            for tip in result["blender_tips"]:
                output_parts.append(f"- {tip}")
                
        # Next steps
        if "next_steps" in result:
            output_parts.append(f"\nNext Steps:")
            for step in result["next_steps"]:
                output_parts.append(f"- {step}")
                
        return "\n".join(output_parts)
        
    async def handle_error(self, error_type: str, context: Dict[str, Any]) -> str:
        """Handle Blender-specific errors with recovery"""
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
