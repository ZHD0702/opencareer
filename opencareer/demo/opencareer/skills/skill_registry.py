"""
SKILL Registry for OpenCareer.

This module provides a registry for managing SKILLs in the OpenCareer system,
supporting dynamic loading, discovery, and execution of SKILLs.
"""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from .base_skill import BaseSkill, SkillMetadata, SkillCategory


class SkillRegistry:
    """Registry for managing SKILLs."""

    def __init__(self):
        self.skills: Dict[str, BaseSkill] = {}
        self.logger = logging.getLogger("skill.registry")

    def register(self, skill: BaseSkill) -> None:
        """Register a SKILL.

        Args:
            skill: The SKILL to register

        Raises:
            ValueError: If a SKILL with the same name is already registered
        """
        skill_name = skill.metadata.name

        if skill_name in self.skills:
            raise ValueError(f"SKILL with name '{skill_name}' is already registered")

        self.skills[skill_name] = skill
        self.logger.info(f"Registered SKILL: {skill_name} (v{skill.metadata.version})")

    def unregister(self, skill_name: str) -> bool:
        """Unregister a SKILL.

        Args:
            skill_name: Name of the SKILL to unregister

        Returns:
            True if SKILL was unregistered, False if not found
        """
        if skill_name in self.skills:
            skill = self.skills.pop(skill_name)
            self.logger.info(f"Unregistered SKILL: {skill_name}")
            return True
        return False

    def get(self, skill_name: str) -> Optional[BaseSkill]:
        """Get a SKILL by name.

        Args:
            skill_name: Name of the SKILL

        Returns:
            The SKILL or None if not found
        """
        return self.skills.get(skill_name)

    def list_skills(self) -> List[Dict[str, Any]]:
        """List all registered SKILLs.

        Returns:
            List of SKILL information dictionaries
        """
        return [
            {
                "name": skill.metadata.name,
                "version": skill.metadata.version,
                "description": skill.metadata.description,
                "category": skill.metadata.category.value,
                "tags": skill.metadata.tags,
                "initialized": skill._initialized
            }
            for skill in self.skills.values()
        ]

    def list_skills_by_category(self, category: Union[SkillCategory, str]) -> List[Dict[str, Any]]:
        """List SKILLs by category.

        Args:
            category: Category to filter by (enum or string)

        Returns:
            List of SKILL information dictionaries
        """
        if isinstance(category, str):
            try:
                category = SkillCategory(category)
            except ValueError:
                # Return empty list for unknown categories
                return []

        return [
            skill_info
            for skill_info in self.list_skills()
            if skill_info["category"] == category.value
        ]

    async def execute(self, skill_name: str, input_data: Dict[str, Any],
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a SKILL.

        Args:
            skill_name: Name of the SKILL to execute
            input_data: Input data for the SKILL
            context: Additional context information

        Returns:
            SKILL execution result

        Raises:
            KeyError: If SKILL is not found
            RuntimeError: If SKILL execution fails
        """
        skill = self.get(skill_name)
        if not skill:
            raise KeyError(f"SKILL not found: {skill_name}")

        # Ensure SKILL is initialized
        if not skill._initialized:
            await skill.initialize()

        try:
            self.logger.info(f"Executing SKILL: {skill_name}")
            result = await skill.execute(input_data, context)
            self.logger.debug(f"SKILL {skill_name} executed successfully")
            return result
        except Exception as e:
            self.logger.error(f"SKILL execution failed: {skill_name}, error: {e}")
            raise RuntimeError(f"SKILL execution failed: {str(e)}")

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all registered SKILLs.

        Returns:
            Health status information
        """
        health_results = {}
        for skill_name, skill in self.skills.items():
            try:
                health = await skill.health_check()
                health_results[skill_name] = health
            except Exception as e:
                health_results[skill_name] = {
                    "status": "error",
                    "error": str(e)
                }

        # Overall status
        healthy_count = sum(
            1 for status in health_results.values()
            if status.get("status") == "healthy"
        )
        total_count = len(health_results)

        return {
            "total_skills": total_count,
            "healthy_skills": healthy_count,
            "unhealthy_skills": total_count - healthy_count,
            "details": health_results,
            "timestamp": datetime.now().isoformat()
        }

    async def initialize_all(self) -> None:
        """Initialize all registered SKILLs."""
        self.logger.info(f"Initializing all SKILLs ({len(self.skills)} total)")

        for skill_name, skill in self.skills.items():
            if not skill._initialized:
                try:
                    await skill.initialize()
                    self.logger.debug(f"Initialized SKILL: {skill_name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize SKILL {skill_name}: {e}")

    async def cleanup_all(self) -> None:
        """Clean up all registered SKILLs."""
        self.logger.info(f"Cleaning up all SKILLs ({len(self.skills)} total)")

        for skill_name, skill in self.skills.items():
            if skill._initialized:
                try:
                    await skill.cleanup()
                    self.logger.debug(f"Cleaned up SKILL: {skill_name}")
                except Exception as e:
                    self.logger.error(f"Failed to clean up SKILL {skill_name}: {e}")

    def discover_skills_in_package(self, package_path: Path) -> List[str]:
        """Discover SKILLs in a Python package.

        Args:
            package_path: Path to the package directory

        Returns:
            List of discovered SKILL names
        """
        discovered = []

        if not package_path.exists():
            self.logger.warning(f"Package path does not exist: {package_path}")
            return discovered

        # Convert to Python module path
        # This is a simplified implementation
        # In a real system, you would need to handle package structure properly
        self.logger.info(f"Discovering SKILLs in: {package_path}")

        # Look for skill.py files
        for skill_dir in package_path.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "skill.py"
                if skill_file.exists():
                    skill_name = skill_dir.name
                    discovered.append(skill_name)
                    self.logger.debug(f"Found SKILL directory: {skill_name}")

        return discovered

    def load_skill_from_module(self, module_path: str, skill_class_name: str = None) -> Optional[BaseSkill]:
        """Load a SKILL from a Python module.

        Args:
            module_path: Path to the Python module
            skill_class_name: Name of the SKILL class (if None, looks for default)

        Returns:
            Loaded SKILL instance or None if failed
        """
        try:
            # Import the module
            module = importlib.import_module(module_path)

            # Find SKILL class
            skill_class = None

            if skill_class_name:
                # Use specified class name
                skill_class = getattr(module, skill_class_name, None)
            else:
                # Look for classes inheriting from BaseSkill
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, BaseSkill) and
                        obj != BaseSkill):
                        skill_class = obj
                        break

            if not skill_class:
                self.logger.error(f"No SKILL class found in module: {module_path}")
                return None

            # Create SKILL instance
            # Assumes the class can be instantiated without arguments
            # or has default values for all parameters
            try:
                skill_instance = skill_class()
                return skill_instance
            except TypeError as e:
                self.logger.error(f"Failed to instantiate SKILL class: {e}")
                return None

        except ImportError as e:
            self.logger.error(f"Failed to import module {module_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading SKILL from module {module_path}: {e}")
            return None


# Global registry instance
_global_registry: Optional[SkillRegistry] = None


def get_global_registry() -> SkillRegistry:
    """Get the global SKILL registry instance.

    Returns:
        Global SKILL registry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry


def register_skill(skill: BaseSkill) -> None:
    """Register a SKILL in the global registry.

    Args:
        skill: The SKILL to register
    """
    registry = get_global_registry()
    registry.register(skill)


async def execute_skill(skill_name: str, input_data: Dict[str, Any],
                       context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute a SKILL from the global registry.

    Args:
        skill_name: Name of the SKILL to execute
        input_data: Input data for the SKILL
        context: Additional context information

    Returns:
        SKILL execution result
    """
    registry = get_global_registry()
    return await registry.execute(skill_name, input_data, context)