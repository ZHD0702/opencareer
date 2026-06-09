"""
Agents module for OpenCareer.

Contains the four core agents:
- Brain (Scheduler Agent)
- Work Agent (Career functions)
- Emotion Agent (Emotional support)
- Log Agent (Information extraction)
"""

from . import brain
from . import work_agent
from . import emotion_agent
from . import log_agent
from .llm_client import LLMClient, create_llm_client

__all__ = [
    "brain",
    "work_agent",
    "emotion_agent",
    "log_agent",
    "LLMClient",
    "create_llm_client",
]
