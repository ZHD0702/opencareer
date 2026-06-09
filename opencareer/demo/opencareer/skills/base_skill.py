"""
Base SKILL class for OpenCareer.

This module provides the base class for all SKILLs in the OpenCareer system,
following the SKILL.md specification format.
"""

import inspect
import logging
import yaml
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable


class SkillCategory(Enum):
    """Categories for SKILLs."""
    LEARNING = "learning"
    INTERVIEW = "interview"
    EMOTION = "emotion"
    UTILITY = "utility"
    DATA = "data"


class SkillMetadata:
    """Metadata for a SKILL following SKILL.md specification."""

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        author: str = "OpenCareer Team",
        category: SkillCategory = SkillCategory.UTILITY,
        tags: List[str] = None,
        input_schema: Dict[str, Any] = None,
        output_schema: Dict[str, Any] = None,
        examples: List[Dict[str, Any]] = None,
        dependencies: List[str] = None
    ):
        """Initialize SKILL metadata.

        Args:
            name: Unique name of the SKILL
            version: Version string
            description: Brief description of the SKILL
            author: Author or maintainer
            category: Category of the SKILL
            tags: List of tags for searching
            input_schema: JSON Schema for input parameters
            output_schema: JSON Schema for output format
            examples: Example usages
            dependencies: Required dependencies
        """
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.category = category
        self.tags = tags or []
        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}
        self.examples = examples or []
        self.dependencies = dependencies or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "category": self.category.value,
            "tags": self.tags,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "examples": self.examples,
            "dependencies": self.dependencies,
            "created_at": datetime.now().isoformat()
        }

    def to_yaml(self) -> str:
        """Convert metadata to YAML string.

        Returns:
            YAML representation
        """
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillMetadata":
        """Create metadata from dictionary.

        Args:
            data: Dictionary containing metadata

        Returns:
            SkillMetadata instance
        """
        # Convert category string to enum
        category_str = data.get("category", "utility")
        try:
            category = SkillCategory(category_str)
        except ValueError:
            category = SkillCategory.UTILITY

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", "OpenCareer Team"),
            category=category,
            tags=data.get("tags", []),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            examples=data.get("examples", []),
            dependencies=data.get("dependencies", [])
        )


class BaseSkill(ABC):
    """Base class for all SKILLs in OpenCareer.

    All SKILLs must inherit from this class and implement the required methods.
    """

    def __init__(self, metadata: SkillMetadata):
        """Initialize the SKILL.

        Args:
            metadata: SKILL metadata
        """
        self.metadata = metadata
        self.logger = logging.getLogger(f"skill.{metadata.name}")
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the SKILL.

        This method should be called before using the SKILL. It can be used
        to load resources, connect to databases, etc.
        """
        if not self._initialized:
            try:
                await self._initialize()
                self._initialized = True
                self.logger.info(f"SKILL {self.metadata.name} initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize SKILL {self.metadata.name}: {e}")
                raise

    async def _initialize(self) -> None:
        """Internal initialization method to be overridden by subclasses."""
        pass

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the SKILL with given input.

        This is the main entry point for SKILL execution.

        Args:
            input_data: Input data matching the input_schema
            context: Additional context information (user ID, session info, etc.)

        Returns:
            Output data matching the output_schema

        Raises:
            ValueError: If input_data doesn't match input_schema
            RuntimeError: If SKILL execution fails
        """
        pass

    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data against input_schema.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid

        Raises:
            ValueError: If input validation fails
        """
        # Basic validation - check required fields
        schema = self.metadata.input_schema
        required_fields = schema.get("required", [])

        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        # Type validation (simplified)
        properties = schema.get("properties", {})
        for field_name, field_value in input_data.items():
            if field_name in properties:
                field_schema = properties[field_name]
                expected_type = field_schema.get("type")

                if expected_type:
                    actual_type = type(field_value).__name__
                    type_mapping = {
                        "str": "string",
                        "int": "integer",
                        "float": "number",
                        "bool": "boolean",
                        "list": "array",
                        "dict": "object"
                    }

                    actual_type_normalized = type_mapping.get(actual_type, actual_type)

                    if actual_type_normalized != expected_type:
                        raise ValueError(
                            f"Field {field_name} has type {actual_type_normalized}, "
                            f"expected {expected_type}"
                        )

        return True

    async def health_check(self) -> Dict[str, Any]:
        """Check health of the SKILL.

        Returns:
            Health status information
        """
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized,
            "timestamp": datetime.now().isoformat()
        }

    async def cleanup(self) -> None:
        """Clean up resources used by the SKILL.

        This method should be called when the SKILL is no longer needed.
        """
        if self._initialized:
            try:
                await self._cleanup()
                self._initialized = False
                self.logger.info(f"SKILL {self.metadata.name} cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up SKILL {self.metadata.name}: {e}")

    async def _cleanup(self) -> None:
        """Internal cleanup method to be overridden by subclasses."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get information about the SKILL.

        Returns:
            SKILL information including metadata
        """
        return {
            "metadata": self.metadata.to_dict(),
            "initialized": self._initialized,
            "class_name": self.__class__.__name__
        }


class FunctionSkill(BaseSkill):
    """SKILL implementation wrapping a Python function.

    This class makes it easy to turn existing functions into SKILLs.
    """

    def __init__(
        self,
        metadata: SkillMetadata,
        func: Callable,
        auto_validate: bool = True
    ):
        """Initialize a function-based SKILL.

        Args:
            metadata: SKILL metadata
            func: The function to wrap
            auto_validate: Whether to automatically validate input
        """
        super().__init__(metadata)
        self.func = func
        self.auto_validate = auto_validate
        self.is_async = inspect.iscoroutinefunction(func)

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the wrapped function.

        Args:
            input_data: Input data for the function
            context: Additional context (passed as keyword argument)

        Returns:
            Function result
        """
        # Validate input if enabled
        if self.auto_validate:
            await self.validate_input(input_data)

        try:
            # Call the function
            if self.is_async:
                if context:
                    result = await self.func(**input_data, context=context)
                else:
                    result = await self.func(**input_data)
            else:
                if context:
                    result = self.func(**input_data, context=context)
                else:
                    result = self.func(**input_data)

            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {"result": result}

            return result

        except Exception as e:
            self.logger.error(f"Error executing function SKILL {self.metadata.name}: {e}")
            raise RuntimeError(f"SKILL execution failed: {str(e)}")

    async def _initialize(self) -> None:
        """Initialize function SKILL (no-op for functions)."""
        pass

    async def _cleanup(self) -> None:
        """Cleanup function SKILL (no-op for functions)."""
        pass


def skill_decorator(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    category: SkillCategory = SkillCategory.UTILITY,
    input_schema: Dict[str, Any] = None,
    output_schema: Dict[str, Any] = None,
    **metadata_kwargs
):
    """Decorator to turn a function into a SKILL.

    Args:
        name: SKILL name
        version: SKILL version
        description: SKILL description
        category: SKILL category
        input_schema: Input JSON schema
        output_schema: Output JSON schema
        **metadata_kwargs: Additional metadata fields

    Returns:
        Decorator function
    """
    def decorator(func):
        # Create metadata
        metadata = SkillMetadata(
            name=name,
            version=version,
            description=description or func.__doc__ or "",
            category=category,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            **metadata_kwargs
        )

        # Create and return SKILL instance
        return FunctionSkill(metadata=metadata, func=func)

    return decorator