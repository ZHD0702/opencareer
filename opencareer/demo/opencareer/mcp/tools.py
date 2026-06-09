"""
LangChain Tools for OpenCareer SKILLs.

This module provides tools for integrating SKILLs with LangChain agents.
"""

import inspect
import logging
from typing import Any, Dict, Optional, Type, Union

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..skills.base_skill import BaseSkill


class SkillToolInput(BaseModel):
    """Input schema for SKILL tools."""
    input_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input data for the SKILL"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context information"
    )


class SkillTool(BaseTool):
    """LangChain tool wrapper for SKILLs."""

    skill: BaseSkill
    skill_name: str

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True

    def _run(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronous tool execution.

        Args:
            input_data: Input data for the SKILL
            context: Additional context information

        Returns:
            SKILL execution result
        """
        # This is a wrapper for synchronous execution
        # In practice, SKILLs are async, so we need to handle this properly
        import asyncio

        async def async_execute():
            return await self.skill.execute(input_data, context or {})

        return asyncio.run(async_execute())

    async def _arun(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Asynchronous tool execution.

        Args:
            input_data: Input data for the SKILL
            context: Additional context information

        Returns:
            SKILL execution result
        """
        return await self.skill.execute(input_data, context or {})

    def get_args_schema(self) -> Type[BaseModel]:
        """Get the input schema for this tool.

        Returns:
            Pydantic model for tool input
        """
        # Create a dynamic schema based on the SKILL's input schema
        skill_input_schema = self.skill.metadata.input_schema

        class DynamicSkillInput(BaseModel):
            """Dynamic input schema based on SKILL metadata."""
            input_data: Dict[str, Any] = Field(
                default_factory=dict,
                description=f"Input data for {self.skill.metadata.name}. Schema: {skill_input_schema}"
            )
            context: Dict[str, Any] = Field(
                default_factory=dict,
                description="Additional context information"
            )

        return DynamicSkillInput


def create_langchain_tool_from_skill(skill: BaseSkill) -> SkillTool:
    """Create a LangChain tool from a SKILL.

    Args:
        skill: The SKILL to wrap

    Returns:
        LangChain tool
    """
    # Create tool name and description
    tool_name = f"skill_{skill.metadata.name}"
    tool_description = f"{skill.metadata.description}"

    # Add category and tags to description
    if skill.metadata.tags:
        tool_description += f"\nTags: {', '.join(skill.metadata.tags)}"

    # Create the tool
    tool = SkillTool(
        name=tool_name,
        description=tool_description,
        skill=skill,
        skill_name=skill.metadata.name,
        args_schema=SkillToolInput
    )

    return tool


def create_tool_from_function(func, name: str = None, description: str = None) -> BaseTool:
    """Create a LangChain tool from a Python function.

    Args:
        func: The function to wrap
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)

    Returns:
        LangChain tool
    """
    from langchain.tools import tool

    # Use function metadata if not provided
    if name is None:
        name = func.__name__
    if description is None:
        description = func.__doc__ or ""

    # Create tool using LangChain's decorator
    @tool(name=name, description=description)
    def wrapped_tool(**kwargs):
        return func(**kwargs)

    return wrapped_tool


def create_async_tool_from_skill(skill: BaseSkill) -> BaseTool:
    """Create an async-compatible LangChain tool from a SKILL.

    Args:
        skill: The SKILL to wrap

    Returns:
        Async-compatible LangChain tool
    """
    # Create tool name and description
    tool_name = f"async_skill_{skill.metadata.name}"
    tool_description = f"Async version of {skill.metadata.description}"

    # Create a custom async tool class
    class AsyncSkillTool(BaseTool):
        """Async-compatible LangChain tool for SKILLs."""

        class Config:
            arbitrary_types_allowed = True

        name: str = tool_name
        description: str = tool_description
        skill: BaseSkill = skill

        def _run(self, *args, **kwargs):
            """Synchronous run - not supported for async tools."""
            raise NotImplementedError("This tool only supports async execution")

        async def _arun(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
            """Async execution."""
            return await self.skill.execute(input_data, context or {})

        def get_args_schema(self) -> Type[BaseModel]:
            """Get input schema."""
            return SkillToolInput

    return AsyncSkillTool()


def create_tool_registry(tools: Dict[str, BaseTool]) -> Dict[str, BaseTool]:
    """Create a registry of tools.

    Args:
        tools: Dictionary of tools

    Returns:
        Tool registry
    """
    registry = {}
    for name, tool in tools.items():
        if not isinstance(tool, BaseTool):
            raise ValueError(f"Tool {name} is not a LangChain tool")
        registry[name] = tool
    return registry


def get_tool_by_name(tools: Dict[str, BaseTool], name: str) -> Optional[BaseTool]:
    """Get a tool by name.

    Args:
        tools: Dictionary of tools
        name: Tool name

    Returns:
        Tool or None if not found
    """
    # Try exact match
    if name in tools:
        return tools[name]

    # Try case-insensitive match
    name_lower = name.lower()
    for tool_name, tool in tools.items():
        if tool_name.lower() == name_lower:
            return tool

    # Try partial match
    for tool_name, tool in tools.items():
        if name_lower in tool_name.lower():
            return tool

    return None


def validate_tool_input(tool: BaseTool, input_data: Dict[str, Any]) -> bool:
    """Validate input data for a tool.

    Args:
        tool: The tool to validate against
        input_data: Input data to validate

    Returns:
        True if input is valid
    """
    try:
        # Get args schema
        args_schema = tool.get_args_schema()

        # Validate input
        validated = args_schema(**input_data)
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(f"Tool input validation failed: {e}")
        return False