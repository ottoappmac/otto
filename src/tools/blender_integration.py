"""Blender-specific tool integration with OTTO framework"""

from typing import Dict, List, Optional, Any
import asyncio
import logging




logger = logging.getLogger(__name__)


class BlenderIntegrationTool:
    """Specialized tool for Blender MCP integration"""
    
    def __init__(self, mcp_client=None):
        self.mcp_client = mcp_client
        self.connection_status = "disconnected"
        
    async def validate_connection(self) -> bool:
        """Validate Blender MCP connection with OTTO's existing patterns"""
        try:
            # Use existing connection monitoring patterns
            if not self.mcp_client:
                return False
            # Test basic tool availability
            response = await self.mcp_client.call_tool("get_scene_info")
            self.connection_status = "connected"
            return True
        except Exception as e:
            logger.warning(f"Blender connection failed: {e}")
            self.connection_status = "error"
            return False
            
    async def execute_blender_code(self, code: str, context: str = "") -> Dict[str, Any]:
        """Execute Blender Python code with OTTO's error handling patterns"""
        try:
            # Pre-validation to prevent crashes
            if not await self.validate_connection():
                raise ConnectionError("Blender connection not available")
                
            # Validate code structure before execution
            if not code.strip():
                raise ValueError("No code provided")
                
            # Execute with timeout to prevent hanging
            result = await asyncio.wait_for(
                self.mcp_client.call_tool("execute_blender_code", {"code": code}),
                timeout=30.0
            )
            
            return {"success": True, "result": result}
            
        except asyncio.TimeoutError:
            return {"success": False, "error": "Blender operation timed out"}
        except Exception as e:
            logger.error(f"Blender execution failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def get_scene_info(self) -> Dict[str, Any]:
        """Get current scene information"""
        try:
            if not await self.validate_connection():
                raise ConnectionError("Blender connection not available")
                
            result = await asyncio.wait_for(
                self.mcp_client.call_tool("get_scene_info"),
                timeout=10.0
            )
            
            return {"success": True, "scene_info": result}
            
        except asyncio.TimeoutError:
            return {"success": False, "error": "Scene info request timed out"}
        except Exception as e:
            logger.error(f"Failed to get scene info: {e}")
            return {"success": False, "error": str(e)}
