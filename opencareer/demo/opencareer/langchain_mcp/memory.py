"""
Memory System for OpenCareer LangChain Agent.

This module provides a unified memory system combining conversation
history, user profile, and long-term memory.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory, CombinedMemory
from langchain.memory.chat_memory import BaseChatMemory

from ..agents.conversation_context import ConversationContext

logger = logging.getLogger(__name__)


class OpenCareerChatHistory(BaseChatMessageHistory):
    """Chat history that integrates with OpenCareer ConversationContext."""

    def __init__(self, context: Optional[ConversationContext] = None):
        """Initialize the chat history.

        Args:
            context: Optional existing ConversationContext
        """
        self.context = context or ConversationContext()
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages."""
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history.

        Args:
            message: The message to add
        """
        self._messages.append(message)

        # Also add to context for consistency
        if isinstance(message, HumanMessage):
            self.context.add_turn("user", message.content)
        elif isinstance(message, AIMessage):
            self.context.add_turn("agent", message.content)

    def add_user_message(self, message: str) -> None:
        """Add a user message.

        Args:
            message: The user message
        """
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Add an AI message.

        Args:
            message: The AI message
        """
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """Clear all messages."""
        self._messages = []
        self.context = ConversationContext()


class OpenCareerMemory:
    """Unified memory system for OpenCareer agent."""

    def __init__(
        self,
        context: Optional[ConversationContext] = None,
        window_size: int = 10,
        memory_file: Optional[Path] = None
    ):
        """Initialize the memory system.

        Args:
            context: Optional existing ConversationContext
            window_size: Number of recent messages to keep in buffer
            memory_file: Optional file for persisting long-term memory
        """
        self.chat_history = OpenCareerChatHistory(context)
        self.window_size = window_size
        self.memory_file = memory_file
        self.long_term_memory: Dict[str, Any] = {}

        # Load existing memory if file exists
        if memory_file and memory_file.exists():
            self._load_memory()

        # Build combined memory
        self._conversation_memory = ConversationBufferWindowMemory(
            chat_memory=self.chat_history,
            k=window_size,
            return_messages=True,
            input_key="input",
            output_key="output"
        )

    @property
    def context(self) -> ConversationContext:
        """Get the conversation context."""
        return self.chat_history.context

    @property
    def langchain_memory(self) -> BaseChatMemory:
        """Get the LangChain memory for agent use."""
        return self._conversation_memory

    def get_user_profile(self) -> Dict[str, Any]:
        """Get the user profile from context.

        Returns:
            User profile dictionary
        """
        return self.context.user_profile or {}

    def update_user_profile(self, **kwargs) -> None:
        """Update the user profile.

        Args:
            **kwargs: Profile fields to update
        """
        if not self.context.user_profile:
            self.context.user_profile = {}
        self.context.user_profile.update(kwargs)
        logger.debug(f"Updated user profile: {kwargs.keys()}")

    def add_to_long_term(self, key: str, value: Any) -> None:
        """Add something to long-term memory.

        Args:
            key: Memory key
            value: Memory value
        """
        self.long_term_memory[key] = value
        logger.debug(f"Added to long-term memory: {key}")

    def get_from_long_term(self, key: str, default: Any = None) -> Any:
        """Get something from long-term memory.

        Args:
            key: Memory key
            default: Default value if key not found

        Returns:
            Memory value or default
        """
        return self.long_term_memory.get(key, default)

    def format_user_profile_for_prompt(self) -> str:
        """Format user profile as a string for prompts.

        Returns:
            Formatted user profile string
        """
        profile = self.get_user_profile()
        if not profile:
            return "Unknown user profile"

        parts = []
        for key, value in profile.items():
            if value:
                parts.append(f"{key}: {value}")

        return "\n".join(parts)

    def get_memory_context(self) -> Dict[str, Any]:
        """Get a full context dictionary for agent use.

        Returns:
            Memory context dictionary
        """
        return {
            "user_profile": self.get_user_profile(),
            "long_term_memory": self.long_term_memory,
            "emotion_history": self.context.emotion_history,
            "mood_history": self.context.mood_history
        }

    def _load_memory(self) -> None:
        """Load memory from file."""
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.long_term_memory = data.get("long_term", {})

                # Load conversation history if available
                if "messages" in data:
                    for msg in data["messages"]:
                        if msg["type"] == "human":
                            self.chat_history.add_user_message(msg["content"])
                        elif msg["type"] == "ai":
                            self.chat_history.add_ai_message(msg["content"])

            logger.info(f"Loaded memory from {self.memory_file}")
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")

    def save_memory(self) -> None:
        """Save memory to file."""
        if not self.memory_file:
            return

        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)

            # Serialize messages
            messages = []
            for msg in self.chat_history.messages:
                if isinstance(msg, HumanMessage):
                    messages.append({"type": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"type": "ai", "content": msg.content})

            data = {
                "long_term": self.long_term_memory,
                "messages": messages,
                "user_profile": self.get_user_profile()
            }

            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved memory to {self.memory_file}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def clear(self) -> None:
        """Clear all memory."""
        self.chat_history.clear()
        self.long_term_memory = {}
        logger.info("Memory cleared")
