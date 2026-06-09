"""
MCP Skill Tool Loader for LangChain.

This module handles loading OpenCareer SKILLs and converting them to
LangChain tools for agent use.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool

from ..skills.skill_registry import get_global_registry, SkillRegistry
from ..skills.base_skill import BaseSkill
from ..mcp.tools import create_langchain_tool_from_skill, SkillTool
from ..mcp.skill_loader import SkillLoader as MCPSkillLoader

logger = logging.getLogger(__name__)


class MCPSkillToolLoader:
    """Loader for converting OpenCareer SKILLs to LangChain tools."""

    def __init__(
        self,
        skills_dir: Optional[Path] = None,
        registry: Optional[SkillRegistry] = None
    ):
        """Initialize the tool loader.

        Args:
            skills_dir: Directory containing SKILLs
            registry: Optional existing SkillRegistry
        """
        self.registry = registry or get_global_registry()
        self.skills_dir = skills_dir or Path(__file__).parent.parent / "skills"
        self.skill_loader = MCPSkillLoader(self.skills_dir)
        self.tools: Dict[str, SkillTool] = {}
        self._loaded = False

    async def load_all_skills(self) -> List[SkillTool]:
        """Load all available SKILLs and convert to tools.

        Returns:
            List of LangChain tools
        """
        logger.info("Loading all SKILLs...")

        # Discover and load SKILLs
        discovered_skills = self.skill_loader.discover_skills()
        logger.info(f"Discovered {len(discovered_skills)} SKILLs")

        for skill_info in discovered_skills:
            try:
                skill = self.skill_loader.load_skill(skill_info["module_path"])
                if skill:
                    await self._register_skill_and_tool(skill)
            except Exception as e:
                logger.error(f"Failed to load skill {skill_info['name']}: {e}")

        # Initialize all skills
        await self.registry.initialize_all()

        self._loaded = True
        return list(self.tools.values())

    async def _register_skill_and_tool(self, skill: BaseSkill) -> None:
        """Register a skill and create its LangChain tool.

        Args:
            skill: The skill to register
        """
        # Register in registry
        self.registry.register(skill)

        # Create LangChain tool
        tool = create_langchain_tool_from_skill(skill)
        self.tools[skill.metadata.name] = tool

        logger.info(f"Registered tool: {tool.name} for skill: {skill.metadata.name}")

    def get_tool(self, skill_name: str) -> Optional[SkillTool]:
        """Get a tool by skill name.

        Args:
            skill_name: Name of the skill

        Returns:
            LangChain tool or None
        """
        return self.tools.get(skill_name)

    def get_all_tools(self) -> List[SkillTool]:
        """Get all loaded tools.

        Returns:
            List of tools
        """
        return list(self.tools.values())

    def get_tools_by_category(self, category: str) -> List[SkillTool]:
        """Get tools by category.

        Args:
            category: Category name

        Returns:
            List of tools in the category
        """
        result = []
        for skill_name, tool in self.tools.items():
            skill = self.registry.get(skill_name)
            if skill and skill.metadata.category.value == category:
                result.append(tool)
        return result

    def get_tools_by_tags(self, tags: List[str]) -> List[SkillTool]:
        """Get tools by tags.

        Args:
            tags: List of tags to match

        Returns:
            List of tools matching any of the tags
        """
        result = []
        for skill_name, tool in self.tools.items():
            skill = self.registry.get(skill_name)
            if skill:
                skill_tags = set(skill.metadata.tags)
                if skill_tags & set(tags):
                    result.append(tool)
        return result

    async def execute_skill_direct(
        self,
        skill_name: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a skill directly without going through agent.

        Args:
            skill_name: Name of the skill
            input_data: Input data for the skill
            context: Additional context

        Returns:
            Skill execution result
        """
        return await self.registry.execute(skill_name, input_data, context)

    async def reload_skills(self) -> List[SkillTool]:
        """Reload all skills.

        Returns:
            List of reloaded tools
        """
        # Cleanup existing
        await self.registry.cleanup_all()
        self.registry = SkillRegistry()
        self.tools.clear()

        # Load fresh
        return await self.load_all_skills()

    def get_available_skill_info(self) -> List[Dict[str, Any]]:
        """Get info about all available skills.

        Returns:
            List of skill info dicts
        """
        return self.registry.list_skills()
