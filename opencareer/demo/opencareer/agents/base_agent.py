"""
Base Agent for OpenCareer multi-agent system.

This module provides a base class for all agents in the OpenCareer system,
following the minimal feature set + extensible architecture pattern.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import dateutil.parser


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication."""
    sender: str
    receiver: str
    content: Dict[str, Any]
    message_type: str = "text"  # text, request, response, error
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(id(object())))

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        from datetime import datetime

        timestamp = self.timestamp
        if isinstance(timestamp, (int, float)):
            # Convert Unix timestamp to ISO format string
            timestamp = datetime.fromtimestamp(timestamp).isoformat()
        elif isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        else:
            # Fallback to string representation
            timestamp = str(timestamp)

        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "message_type": self.message_type,
            "timestamp": timestamp,
            "message_id": self.message_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create an AgentMessage from a dictionary.

        Args:
            data: Dictionary representation of the message

        Returns:
            AgentMessage instance
        """
        from datetime import datetime

        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = dateutil.parser.parse(timestamp)
            except:
                # Fallback to current time
                timestamp = datetime.now()
        elif isinstance(timestamp, (int, float)):
            # Handle Unix timestamp
            timestamp = datetime.fromtimestamp(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            message_type=data.get("message_type", "text"),
            timestamp=timestamp,
            message_id=data.get("message_id", str(id(object())))
        )


class BaseAgent(ABC):
    """Base class for all OpenCareer agents.

    Attributes:
        name (str): Unique name of the agent
        description (str): Brief description of agent's purpose
        capabilities (List[str]): List of capabilities this agent provides
        logger (logging.Logger): Agent-specific logger
        message_queue (asyncio.Queue): Queue for incoming messages
        is_running (bool): Whether the agent is currently running
    """

    def __init__(self, name: str, description: str = ""):
        """Initialize the base agent.

        Args:
            name: Unique name for this agent
            description: Brief description of agent's purpose
        """
        self.name = name
        self.description = description
        self.capabilities: List[str] = []
        self.logger = logging.getLogger(f"agent.{name}")
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.mailbox = self.message_queue  # Alias for test compatibility
        self.is_running = False
        self._message_handlers: Dict[str, callable] = {}

        # Register default message handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.register_message_handler("ping", self._handle_ping)
        self.register_message_handler("status", self._handle_status_request)

    def register_message_handler(self, message_type: str, handler: callable) -> None:
        """Register a handler for a specific message type.

        Args:
            message_type: Type of message to handle
            handler: Function to handle the message
        """
        self._message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for message type: {message_type}")

    def send_message(self, message_or_receiver, content_or_agent=None, message_type: str = "text") -> AgentMessage:
        """Send a message to another agent.

        Supports two calling conventions:
        1. send_message(receiver: str, content: Dict[str, Any], message_type: str = "text")
        2. send_message(message: AgentMessage, receiver_agent: BaseAgent)  # For test compatibility

        Args:
            message_or_receiver: Either receiver name (str) or AgentMessage object
            content_or_agent: Either message content (Dict) or receiver agent (BaseAgent)
            message_type: Type of message (used with convention 1)

        Returns:
            The sent message
        """
        # Check which calling convention is being used
        # Import here to avoid circular import
        from .base_agent import BaseAgent as BaseAgentType

        if isinstance(message_or_receiver, AgentMessage) and isinstance(content_or_agent, BaseAgentType):
            # Convention 2: send_message(message, receiver_agent)
            message = message_or_receiver
            receiver_agent = content_or_agent

            # In a real implementation, this would route through a message bus
            # For now, we'll just call receive_message on the receiver
            self.logger.info(f"Sending message to {receiver_agent.name}: {message.message_type}")
            self.logger.debug(f"Message content: {message.content}")

            # For test compatibility, schedule the receive
            import asyncio
            asyncio.create_task(receiver_agent.receive_message(message))

            return message
        else:
            # Convention 1: send_message(receiver, content, message_type)
            receiver = message_or_receiver
            content = content_or_agent

            message = AgentMessage(
                sender=self.name,
                receiver=receiver,
                content=content,
                message_type=message_type
            )

            # In a real implementation, this would route through a message bus
            # For now, we'll log the message
            self.logger.info(f"Sending message to {receiver}: {message_type}")
            self.logger.debug(f"Message content: {content}")

            # TODO: Implement actual message routing
            return message

    async def receive_message(self, message: AgentMessage) -> None:
        """Receive and process a message.

        Args:
            message: The message to process
        """
        self.logger.info(f"Received message from {message.sender}: {message.message_type}")

        # Add message to queue for processing
        await self.message_queue.put(message)

        # Process the message if agent is running
        if self.is_running:
            asyncio.create_task(self._process_message_queue())

    async def _process_message_queue(self) -> None:
        """Process messages from the queue."""
        while not self.message_queue.empty():
            try:
                message = await self.message_queue.get()
                await self._handle_message(message)
                self.message_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")

    async def _handle_message(self, message: AgentMessage) -> None:
        """Handle a single message.

        Args:
            message: The message to handle
        """
        handler = self._message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                self.logger.error(f"Error in message handler for {message.message_type}: {e}")
                # Send error response
                error_response = AgentMessage(
                    sender=self.name,
                    receiver=message.sender,
                    content={"error": str(e), "original_message_id": message.message_id},
                    message_type="error"
                )
                # TODO: Send the error response
        else:
            self.logger.warning(f"No handler for message type: {message.message_type}")
            # Try to handle as text if no specific handler
            if message.message_type == "text":
                await self.handle_text_message(message)

    async def _handle_ping(self, message: AgentMessage) -> None:
        """Handle ping messages."""
        self.logger.debug(f"Received ping from {message.sender}")
        # Send pong response
        pong_message = AgentMessage(
            sender=self.name,
            receiver=message.sender,
            content={"status": "alive", "agent": self.name},
            message_type="pong"
        )
        # TODO: Send pong message

    async def _handle_status_request(self, message: AgentMessage) -> None:
        """Handle status request messages."""
        status_info = {
            "agent": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "is_running": self.is_running,
            "queue_size": self.message_queue.qsize()
        }

        status_message = AgentMessage(
            sender=self.name,
            receiver=message.sender,
            content=status_info,
            message_type="status_response"
        )
        # TODO: Send status message

    async def handle_text_message(self, message: AgentMessage) -> None:
        """Handle text messages. To be overridden by subclasses.

        Args:
            message: The text message to handle
        """
        self.logger.info(f"Received text message from {message.sender}: {message.content}")
        # Base implementation does nothing - subclasses should override

    async def start(self) -> None:
        """Start the agent's main loop."""
        if self.is_running:
            self.logger.warning(f"Agent {self.name} is already running")
            return

        self.is_running = True
        self.logger.info(f"Starting agent: {self.name}")

        # Start message processing loop
        asyncio.create_task(self._agent_loop())

    async def stop(self) -> None:
        """Stop the agent."""
        self.is_running = False
        self.logger.info(f"Stopping agent: {self.name}")

    async def _agent_loop(self) -> None:
        """Main agent loop for background processing."""
        while self.is_running:
            try:
                # Process any pending messages
                await self._process_message_queue()

                # Do agent-specific background work
                await self.background_work()

                # Sleep to prevent busy looping
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in agent loop: {e}")
                await asyncio.sleep(1)  # Sleep on error

    async def background_work(self) -> None:
        """Perform background work. To be overridden by subclasses."""
        pass

    async def process_user_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user request. Main entry point for agent functionality.

        Args:
            user_input: The user's input text
            context: Additional context for processing

        Returns:
            Response from the agent
        """
        self.logger.info(f"Processing user request: {user_input[:50]}...")

        # Default implementation - subclasses should override
        return {
            "agent": self.name,
            "response": "This agent hasn't implemented request processing yet.",
            "status": "not_implemented"
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the agent.

        Returns:
            Status information
        """
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "is_running": self.is_running,
            "queue_size": self.message_queue.qsize(),
            "handlers_registered": list(self._message_handlers.keys())
        }


class AgentRegistry:
    """Registry for managing agents in the system."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent.

        Args:
            agent: The agent to register
        """
        if agent.name in self.agents:
            raise ValueError(f"Agent with name {agent.name} already registered")

        self.agents[agent.name] = agent
        logging.info(f"Registered agent: {agent.name}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name.

        Args:
            name: Name of the agent

        Returns:
            The agent or None if not found
        """
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names.

        Returns:
            List of agent names
        """
        return list(self.agents.keys())

    async def broadcast_message(self, sender: str, content: Dict[str, Any],
                               message_type: str = "text", exclude_sender: bool = True) -> None:
        """Broadcast a message to all agents.

        Args:
            sender: Name of the sending agent
            content: Message content
            message_type: Type of message
            exclude_sender: Whether to exclude the sender from receiving the message
        """
        message = AgentMessage(
            sender=sender,
            receiver="broadcast",
            content=content,
            message_type=message_type
        )

        for agent_name, agent in self.agents.items():
            if exclude_sender and agent_name == sender:
                continue

            try:
                await agent.receive_message(message)
            except Exception as e:
                logging.error(f"Error broadcasting to agent {agent_name}: {e}")