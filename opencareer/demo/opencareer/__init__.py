"""
OpenCareer - Multi-agent Career Assistant

A demo implementation of a multi-agent career assistant using LangChain + MCP + SKILLs architecture.
"""

__version__ = "0.1.0"
__author__ = "OpenCareer Team"
__description__ = "Multi-agent architecture for career assistance"

from . import agents
from . import skills
from . import mcp
from . import memory
from . import scheduler
from . import utils

__all__ = [
    "agents",
    "skills",
    "mcp",
    "memory",
    "scheduler",
    "utils",
]