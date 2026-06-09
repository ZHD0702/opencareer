"""
LangChain + MCP Integration for OpenCareer.

This module provides seamless integration between LangChain agents and
OpenCareer SKILLs via MCP protocol.
"""

from .agent import OpenCareerAgent, create_opencareer_agent
from .tool_loader import MCPSkillToolLoader
from .memory import OpenCareerMemory

__all__ = [
    "OpenCareerAgent",
    "create_opencareer_agent",
    "MCPSkillToolLoader",
    "OpenCareerMemory"
]
