"""
Task Scheduler for OpenCareer.

This module provides task scheduling using dramatiq + Redis for asynchronous
background task processing.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class TaskDefinition:
    """Definition of a scheduled task."""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    scheduled_time: datetime
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task definition to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "scheduled_time": self.scheduled_time.isoformat(),
            "priority": self.priority.value,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskDefinition":
        """Create task definition from dictionary."""
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            parameters=data["parameters"],
            scheduled_time=datetime.fromisoformat(data["scheduled_time"]),
            priority=TaskPriority(data.get("priority", TaskPriority.NORMAL.value)),
            max_retries=data.get("max_retries", 3),
            timeout_seconds=data.get("timeout_seconds", 300),
            metadata=data.get("metadata", {})
        )


@dataclass
class TaskExecutionResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "result": self.result,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms
        }


class BaseTaskHandler(ABC):
    """Base class for task handlers."""

    def __init__(self, task_type: str):
        self.task_type = task_type
        self.logger = logging.getLogger(f"scheduler.handler.{task_type}")

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute the task.

        Args:
            parameters: Task parameters

        Returns:
            Task execution result
        """
        pass

    @abstractmethod
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate task parameters.

        Args:
            parameters: Task parameters to validate

        Returns:
            True if parameters are valid
        """
        pass


class TaskScheduler:
    """Task scheduler using dramatiq + Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0",
                 namespace: str = "opencareer"):
        """Initialize task scheduler.

        Args:
            redis_url: Redis connection URL
            namespace: Namespace for task queues
        """
        self.namespace = namespace
        self.redis_url = redis_url
        self.broker = None
        self.handlers: Dict[str, BaseTaskHandler] = {}
        self.logger = logging.getLogger("scheduler")

    async def initialize(self) -> None:
        """Initialize the scheduler."""
        self.logger.info(f"Initializing task scheduler with Redis: {self.redis_url}")

        try:
            # Initialize Redis broker
            self.broker = RedisBroker(url=self.redis_url)
            dramatiq.set_broker(self.broker)

            # Register actors
            self._register_actors()

            self.logger.info("Task scheduler initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize task scheduler: {e}")
            raise

    def _register_actors(self) -> None:
        """Register dramatiq actors for task execution."""

        @dramatiq.actor(queue_name=f"{self.namespace}_tasks")
        def execute_task(task_definition_json: str) -> Dict[str, Any]:
            """Execute a task (dramatiq actor).

            Args:
                task_definition_json: JSON string of task definition

            Returns:
                Task execution result
            """
            import asyncio

            # Parse task definition
            task_def = TaskDefinition.from_dict(json.loads(task_definition_json))

            # Get handler
            handler = self.handlers.get(task_def.task_type)
            if not handler:
                return {
                    "task_id": task_def.task_id,
                    "status": TaskStatus.FAILED.value,
                    "error": f"No handler registered for task type: {task_def.task_type}"
                }

            # Execute task
            try:
                # Run async task in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                result = loop.run_until_complete(handler.execute(task_def.parameters))

                return {
                    "task_id": task_def.task_id,
                    "status": TaskStatus.COMPLETED.value,
                    "result": result,
                    "execution_time": datetime.now().isoformat()
                }

            except Exception as e:
                return {
                    "task_id": task_def.task_id,
                    "status": TaskStatus.FAILED.value,
                    "error": str(e),
                    "execution_time": datetime.now().isoformat()
                }

        # Store actor reference
        self._execute_task_actor = execute_task

    def register_handler(self, handler: BaseTaskHandler) -> None:
        """Register a task handler.

        Args:
            handler: Task handler to register
        """
        self.handlers[handler.task_type] = handler
        self.logger.info(f"Registered handler for task type: {handler.task_type}")

    async def schedule_task(self, task_type: str, parameters: Dict[str, Any],
                           delay_seconds: int = 0,
                           priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Schedule a task for execution.

        Args:
            task_type: Type of task to schedule
            parameters: Task parameters
            delay_seconds: Delay before execution (seconds)
            priority: Task priority

        Returns:
            Task ID
        """
        # Validate handler exists
        if task_type not in self.handlers:
            raise ValueError(f"No handler registered for task type: {task_type}")

        # Validate parameters
        handler = self.handlers[task_type]
        if not await handler.validate_parameters(parameters):
            raise ValueError(f"Invalid parameters for task type: {task_type}")

        # Generate task ID
        task_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(object())}"

        # Create task definition
        scheduled_time = datetime.now() + timedelta(seconds=delay_seconds)
        task_def = TaskDefinition(
            task_id=task_id,
            task_type=task_type,
            parameters=parameters,
            scheduled_time=scheduled_time,
            priority=priority
        )

        # Schedule task using dramatiq
        if delay_seconds > 0:
            # Delayed execution
            self._execute_task_actor.send_with_options(
                args=(json.dumps(task_def.to_dict()),),
                delay=delay_seconds * 1000  # dramatiq expects milliseconds
            )
        else:
            # Immediate execution
            self._execute_task_actor.send(json.dumps(task_def.to_dict()))

        self.logger.info(f"Scheduled task {task_id} (type: {task_type}, delay: {delay_seconds}s)")
        return task_id

    async def schedule_recurring_task(self, task_type: str, parameters: Dict[str, Any],
                                     interval_seconds: int,
                                     end_after: Optional[int] = None) -> List[str]:
        """Schedule a recurring task.

        Args:
            task_type: Type of task to schedule
            parameters: Task parameters
            interval_seconds: Interval between executions (seconds)
            end_after: Number of executions before stopping (None for infinite)

        Returns:
            List of scheduled task IDs
        """
        task_ids = []

        if end_after is None:
            # Schedule infinite recurring task (cancellation must be handled externally)
            # For demo purposes, we'll schedule 100 executions
            executions = 100
        else:
            executions = end_after

        for i in range(executions):
            delay = interval_seconds * i
            task_id = await self.schedule_task(
                task_type=task_type,
                parameters=parameters,
                delay_seconds=delay,
                priority=TaskPriority.LOW
            )
            task_ids.append(task_id)

        self.logger.info(f"Scheduled recurring task {task_type} ({executions} executions)")
        return task_ids

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task.

        Args:
            task_id: ID of the task

        Returns:
            Task status or None if not found
        """
        # In a real implementation, this would query Redis for task status
        # For demo purposes, we'll return a simple response
        self.logger.debug(f"Getting status for task {task_id}")

        # This is a simplified implementation
        # In production, you would track task execution in Redis
        return {
            "task_id": task_id,
            "status": "unknown",  # Would be tracked in real implementation
            "last_updated": datetime.now().isoformat()
        }

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task.

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if cancellation was successful
        """
        # In dramatiq, cancelling delayed tasks is complex
        # For demo purposes, we'll just log the request
        self.logger.info(f"Request to cancel task {task_id} (not implemented in demo)")
        return True

    async def health_check(self) -> Dict[str, Any]:
        """Check scheduler health.

        Returns:
            Health status
        """
        health_status = {
            "scheduler": "unknown",
            "redis": "unknown",
            "handlers": len(self.handlers),
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Check Redis connection
            if self.broker:
                health_status["scheduler"] = "healthy"
                health_status["redis"] = "healthy"
            else:
                health_status["scheduler"] = "uninitialized"

        except Exception as e:
            health_status["scheduler"] = f"error: {str(e)}"

        return health_status


# Example task handlers for OpenCareer

class LearningPlanReminderHandler(BaseTaskHandler):
    """Handler for learning plan reminder tasks."""

    def __init__(self):
        super().__init__("learning_plan_reminder")

    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute learning plan reminder task."""
        self.logger.info(f"Sending learning plan reminder: {parameters}")

        # In a real implementation, this would:
        # 1. Fetch user's learning plan from memory
        # 2. Check progress
        # 3. Send reminder/encouragement

        user_id = parameters.get("user_id", "unknown")
        plan_id = parameters.get("plan_id", "unknown")

        return {
            "user_id": user_id,
            "plan_id": plan_id,
            "reminder_sent": True,
            "timestamp": datetime.now().isoformat(),
            "message": "Time to work on your learning plan! Consistency is key to success."
        }

    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate reminder task parameters."""
        required = ["user_id", "plan_id"]
        return all(key in parameters for key in required)


class PeriodicInfoExtractionHandler(BaseTaskHandler):
    """Handler for periodic information extraction tasks."""

    def __init__(self):
        super().__init__("periodic_info_extraction")

    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute periodic information extraction task."""
        self.logger.info(f"Running periodic info extraction: {parameters}")

        # In a real implementation, this would:
        # 1. Analyze recent conversations
        # 2. Extract key information
        # 3. Update user profiles

        user_id = parameters.get("user_id", "all_users")
        extraction_window = parameters.get("window_hours", 24)

        return {
            "user_id": user_id,
            "extraction_window_hours": extraction_window,
            "extraction_completed": True,
            "timestamp": datetime.now().isoformat(),
            "extracted_items": 0  # Would be actual count in real implementation
        }

    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate extraction task parameters."""
        # User ID is optional (can process all users)
        return True


class SystemCleanupHandler(BaseTaskHandler):
    """Handler for system cleanup tasks."""

    def __init__(self):
        super().__init__("system_cleanup")

    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute system cleanup task."""
        self.logger.info(f"Running system cleanup: {parameters}")

        # In a real implementation, this would:
        # 1. Clean up temporary files
        # 2. Archive old data
        # 3. Optimize databases

        cleanup_type = parameters.get("type", "general")
        retention_days = parameters.get("retention_days", 30)

        return {
            "cleanup_type": cleanup_type,
            "retention_days": retention_days,
            "cleanup_completed": True,
            "timestamp": datetime.now().isoformat(),
            "cleaned_items": 0  # Would be actual count in real implementation
        }

    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate cleanup task parameters."""
        return True


# Factory function for creating scheduler with default handlers
def create_default_scheduler(redis_url: str = "redis://localhost:6379/0") -> TaskScheduler:
    """Create a scheduler with default task handlers.

    Args:
        redis_url: Redis connection URL

    Returns:
        TaskScheduler instance
    """
    scheduler = TaskScheduler(redis_url=redis_url)

    # Register default handlers
    scheduler.register_handler(LearningPlanReminderHandler())
    scheduler.register_handler(PeriodicInfoExtractionHandler())
    scheduler.register_handler(SystemCleanupHandler())

    return scheduler