"""
Memory system for OpenCareer.

Implements hybrid storage:
- Vector memory (ChromaDB) for conversation history
- Structured memory (SQLite/JSON) for user data
"""

from .memory_manager import (
    BaseMemory,
    MemoryManager,
    MemoryQuery,
    MemoryRecord,
    MemorySearchResult,
)
from .resume_store import ResumeStore, create_resume_store
from .structured_memory import JSONStructuredMemory, SQLiteStructuredMemory, create_structured_memory

__all__ = [
    # Base interfaces
    "BaseMemory",
    "MemoryManager",
    "MemoryQuery",
    "MemoryRecord",
    "MemorySearchResult",
    # Resume store
    "ResumeStore",
    "create_resume_store",
    # Structured memory backends
    "JSONStructuredMemory",
    "SQLiteStructuredMemory",
    "create_structured_memory",
]