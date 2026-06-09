"""
Memory Manager for OpenCareer.

This module provides a unified interface for managing both vector-based
and structured memory storage in the OpenCareer system.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field


@dataclass
class MemoryRecord:
    """Base class for memory records."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    memory_id: str = field(default_factory=lambda: str(id(object())))

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "memory_id": self.memory_id
        }


class MemoryQuery(BaseModel):
    """Query for searching memories."""
    text: Optional[str] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    limit: int = 10
    similarity_threshold: float = 0.7


class MemorySearchResult(BaseModel):
    """Result of a memory search."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None
    timestamp: datetime
    source_id: str = ""  # memory_id of the source record, for update/delete operations


class BaseMemory(ABC):
    """Base class for memory storage implementations."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"memory.{name}")

    @abstractmethod
    async def store(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Store content in memory.

        Args:
            content: The content to store
            metadata: Additional metadata

        Returns:
            ID of the stored memory
        """
        pass

    @abstractmethod
    async def search(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """Search for memories matching the query.

        Args:
            query: The search query

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memories."""
        pass

    async def update(self, memory_id: str, content: str = None,
                    metadata: Dict[str, Any] = None) -> bool:
        """Update a memory.

        Args:
            memory_id: ID of the memory to update
            content: New content (if provided)
            metadata: New metadata (if provided)

        Returns:
            True if update was successful
        """
        # Default implementation: delete and re-create
        # Subclasses can override with more efficient implementations
        try:
            # First, retrieve the existing memory
            existing = await self.get(memory_id)
            if existing is None:
                return False

            # Merge metadata
            new_metadata = existing.metadata.copy()
            if metadata:
                new_metadata.update(metadata)

            # Use existing content if not provided
            new_content = content if content is not None else existing.content

            # Delete old memory
            await self.delete(memory_id)

            # Store new memory
            await self.store(new_content, new_metadata)
            return True

        except Exception as e:
            self.logger.error(f"Error updating memory {memory_id}: {e}")
            return False

    @abstractmethod
    async def get(self, memory_id: str) -> Optional[MemorySearchResult]:
        """Get a memory by ID.

        Args:
            memory_id: ID of the memory

        Returns:
            The memory or None if not found
        """
        pass


class MemoryManager:
    """Manager for hybrid memory storage (vector + structured)."""

    def __init__(self, vector_memory: BaseMemory, structured_memory: BaseMemory):
        """Initialize memory manager.

        Args:
            vector_memory: Vector-based memory storage
            structured_memory: Structured memory storage
        """
        self.vector_memory = vector_memory
        self.structured_memory = structured_memory
        self.logger = logging.getLogger("memory.manager")

    async def store_conversation(self, user_id: str, conversation: Dict[str, Any]) -> str:
        """Store a conversation in both vector and structured memory.

        Args:
            user_id: ID of the user
            conversation: Conversation data

        Returns:
            ID of the stored conversation
        """
        # Extract text for vector storage
        conversation_text = self._extract_conversation_text(conversation)

        # Store in vector memory with metadata
        vector_metadata = {
            "user_id": user_id,
            "type": "conversation",
            "timestamp": datetime.now().isoformat(),
            "source": "conversation"
        }

        vector_id = await self.vector_memory.store(
            content=conversation_text,
            metadata={**vector_metadata, **conversation.get("metadata", {})}
        )

        # Store structured data
        structured_metadata = {
            "user_id": user_id,
            "vector_id": vector_id,
            "conversation_length": len(conversation.get("messages", [])),
            "participants": conversation.get("participants", ["user", "assistant"])
        }

        structured_content = json.dumps({
            "conversation": conversation,
            "metadata": structured_metadata
        })

        await self.structured_memory.store(
            content=structured_content,
            metadata=structured_metadata
        )

        self.logger.info(f"Stored conversation for user {user_id}, vector_id: {vector_id}")
        return vector_id

    async def store_user_profile(self, user_id: str, profile: Dict[str, Any]) -> str:
        """Store user profile data.

        Args:
            user_id: ID of the user
            profile: Profile data

        Returns:
            ID of the stored profile
        """
        # Store in structured memory
        metadata = {
            "user_id": user_id,
            "type": "user_profile",
            "timestamp": datetime.now().isoformat()
        }

        content = json.dumps(profile)

        profile_id = await self.structured_memory.store(
            content=content,
            metadata=metadata
        )

        # Also store key profile information in vector memory for semantic search
        profile_text = self._extract_profile_text(profile)
        if profile_text:
            await self.vector_memory.store(
                content=profile_text,
                metadata={**metadata, "structured_id": profile_id}
            )

        self.logger.info(f"Stored profile for user {user_id}")
        return profile_id

    async def store_skill(self, user_id: str, skill_data: Dict[str, Any]) -> str:
        """Store skill information.

        Args:
            user_id: ID of the user
            skill_data: Skill data

        Returns:
            ID of the stored skill
        """
        metadata = {
            "user_id": user_id,
            "type": "skill",
            "timestamp": datetime.now().isoformat(),
            "skill_name": skill_data.get("name", "unknown")
        }

        content = json.dumps(skill_data)

        skill_id = await self.structured_memory.store(
            content=content,
            metadata=metadata
        )

        # Store skill description in vector memory
        skill_description = skill_data.get("description", "")
        if skill_description:
            await self.vector_memory.store(
                content=skill_description,
                metadata={**metadata, "structured_id": skill_id}
            )

        return skill_id

    async def search_conversations(self, user_id: str, query_text: str = None,
                                  limit: int = 10) -> List[MemorySearchResult]:
        """Search for conversations.

        Args:
            user_id: ID of the user
            query_text: Text to search for
            limit: Maximum number of results

        Returns:
            List of search results
        """
        if query_text:
            # Use vector search
            search_query = MemoryQuery(
                text=query_text,
                metadata_filter={"user_id": user_id, "type": "conversation"},
                limit=limit
            )
            return await self.vector_memory.search(search_query)
        else:
            # Get recent conversations
            search_query = MemoryQuery(
                metadata_filter={"user_id": user_id, "type": "conversation"},
                limit=limit
            )
            return await self.structured_memory.search(search_query)

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile.

        Args:
            user_id: ID of the user

        Returns:
            User profile or None if not found
        """
        search_query = MemoryQuery(
            metadata_filter={"user_id": user_id, "type": "user_profile"},
            limit=1
        )

        results = await self.structured_memory.search(search_query)
        if results:
            try:
                return json.loads(results[0].content)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse profile for user {user_id}")
                return None

        return None

    async def get_user_skills(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user skills.

        Args:
            user_id: ID of the user

        Returns:
            List of user skills
        """
        search_query = MemoryQuery(
            metadata_filter={"user_id": user_id, "type": "skill"},
            limit=100
        )

        results = await self.structured_memory.search(search_query)
        skills = []

        for result in results:
            try:
                skill_data = json.loads(result.content)
                skills.append(skill_data)
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse skill data: {result.content}")

        return skills

    async def clear_user_data(self, user_id: str) -> None:
        """Clear all data for a user.

        Args:
            user_id: ID of the user
        """
        # This is a simplified implementation
        # In a real system, you would need to handle this more carefully
        self.logger.info(f"Clearing data for user {user_id}")

        # Note: Actual implementation would require deleting from both stores
        # with proper filtering by user_id
        raise NotImplementedError("clear_user_data not implemented yet")

    def _extract_conversation_text(self, conversation: Dict[str, Any]) -> str:
        """Extract text from conversation for vector storage.

        Args:
            conversation: Conversation data

        Returns:
            Extracted text
        """
        messages = conversation.get("messages", [])
        text_parts = []

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                text_parts.append(f"{role}: {content}")
            elif isinstance(msg, str):
                text_parts.append(msg)

        return "\n".join(text_parts)

    def _extract_profile_text(self, profile: Dict[str, Any]) -> str:
        """Extract text from profile for vector storage.

        Args:
            profile: Profile data

        Returns:
            Extracted text
        """
        text_parts = []

        # Add key profile fields
        for field in ["name", "title", "summary", "experience", "education"]:
            if field in profile:
                value = profile[field]
                if isinstance(value, str):
                    text_parts.append(f"{field}: {value}")
                elif isinstance(value, list):
                    text_parts.append(f"{field}: {', '.join(str(v) for v in value)}")

        return "\n".join(text_parts)

    async def health_check(self) -> Dict[str, Any]:
        """Check health of memory systems.

        Returns:
            Health status
        """
        health_status = {
            "vector_memory": "unknown",
            "structured_memory": "unknown",
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Simple test for vector memory
            test_id = await self.vector_memory.store(
                content="health_check",
                metadata={"type": "health_check", "timestamp": datetime.now().isoformat()}
            )
            await self.vector_memory.delete(test_id)
            health_status["vector_memory"] = "healthy"
        except Exception as e:
            health_status["vector_memory"] = f"error: {str(e)}"

        try:
            # Simple test for structured memory
            test_id = await self.structured_memory.store(
                content="health_check",
                metadata={"type": "health_check", "timestamp": datetime.now().isoformat()}
            )
            await self.structured_memory.delete(test_id)
            health_status["structured_memory"] = "healthy"
        except Exception as e:
            health_status["structured_memory"] = f"error: {str(e)}"

        return health_status