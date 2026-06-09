"""
SKILL Loader for MCP Server.

This module handles dynamic loading and discovery of SKILLs for the MCP Server.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..skills.base_skill import BaseSkill
from ..skills.skill_registry import SkillMetadata


class SkillLoader:
    """Loader for dynamically discovering and loading SKILLs."""

    def __init__(self, skills_dir: Path):
        """Initialize SKILL loader.

        Args:
            skills_dir: Directory containing SKILLs
        """
        self.skills_dir = skills_dir
        self.logger = logging.getLogger("mcp.skill_loader")

    def discover_skills(self) -> List[Dict[str, Any]]:
        """Discover SKILLs in the skills directory.

        Returns:
            List of discovered SKILL information
        """
        discovered_skills = []

        if not self.skills_dir.exists():
            self.logger.warning(f"Skills directory does not exist: {self.skills_dir}")
            return discovered_skills

        self.logger.info(f"Discovering SKILLs in: {self.skills_dir}")

        # Look for skill directories
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                # Check for skill.py file
                skill_file = skill_dir / "skill.py"
                if skill_file.exists():
                    # Try to extract SKILL information
                    skill_info = self._extract_skill_info(skill_dir)
                    if skill_info:
                        discovered_skills.append(skill_info)
                        self.logger.debug(f"Found SKILL: {skill_info['name']}")

        self.logger.info(f"Discovered {len(discovered_skills)} SKILLs")
        return discovered_skills

    def _extract_skill_info(self, skill_dir: Path) -> Optional[Dict[str, Any]]:
        """Extract SKILL information from directory.

        Args:
            skill_dir: Directory containing the SKILL

        Returns:
            SKILL information or None if extraction failed
        """
        try:
            # Convert path to module path
            # Assumes structure: skills/<skill_name>/skill.py
            skill_name = skill_dir.name
            module_path = f"..skills.{skill_name}.skill"

            return {
                "name": skill_name,
                "module_path": module_path,
                "directory": skill_dir
            }
        except Exception as e:
            self.logger.error(f"Failed to extract SKILL info from {skill_dir}: {e}")
            return None

    def load_skill(self, module_path: str) -> Optional[BaseSkill]:
        """Load a SKILL from module path.

        Args:
            module_path: Path to the SKILL module

        Returns:
            Loaded SKILL instance or None if loading failed
        """
        try:
            # Remove leading dots if present
            if module_path.startswith(".."):
                # Convert to absolute module path
                parts = module_path.split(".")
                if len(parts) >= 4:
                    # Format: ..skills.<skill_name>.skill
                    # split: ['', '', 'skills', '<skill_name>', 'skill']
                    skill_name = parts[3]
                    module_path = f"opencareer.skills.{skill_name}.skill"
                else:
                    self.logger.error(f"Invalid module path format: {module_path}")
                    return None

            # Import module
            module = importlib.import_module(module_path)

            # Find SKILL class (subclass of BaseSkill)
            skill_class = None
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
            # SKILL classes should have a default constructor
            try:
                skill_instance = skill_class()
                self.logger.info(f"Loaded SKILL: {skill_instance.metadata.name}")
                return skill_instance
            except TypeError as e:
                self.logger.error(f"Failed to instantiate SKILL class: {e}")
                return None

        except ImportError as e:
            self.logger.error(f"Failed to import module {module_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading SKILL from {module_path}: {e}")
            return None

    def load_skill_from_module(self, module_path: str, skill_class_name: str = None) -> Optional[BaseSkill]:
        """Load a SKILL from a specific module and class.

        Args:
            module_path: Full module path
            skill_class_name: Name of the SKILL class (optional)

        Returns:
            Loaded SKILL instance or None if loading failed
        """
        try:
            # Import module
            module = importlib.import_module(module_path)

            # Find SKILL class
            skill_class = None
            if skill_class_name:
                # Use specified class name
                skill_class = getattr(module, skill_class_name, None)
            else:
                # Find first subclass of BaseSkill
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
            try:
                skill_instance = skill_class()
                self.logger.info(f"Loaded SKILL: {skill_instance.metadata.name}")
                return skill_instance
            except TypeError as e:
                self.logger.error(f"Failed to instantiate SKILL class: {e}")
                return None

        except ImportError as e:
            self.logger.error(f"Failed to import module {module_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading SKILL from {module_path}: {e}")
            return None

    def load_skill_from_directory(self, skill_dir: Path) -> Optional[BaseSkill]:
        """Load a SKILL from a directory.

        Args:
            skill_dir: Directory containing the SKILL

        Returns:
            Loaded SKILL instance or None if loading failed
        """
        # Check for skill.py
        skill_file = skill_dir / "skill.py"
        if not skill_file.exists():
            self.logger.error(f"No skill.py found in directory: {skill_dir}")
            return None

        # Try to load from module path
        skill_name = skill_dir.name
        module_path = f"opencareer.skills.{skill_name}.skill"

        return self.load_skill(module_path)

    def create_skill_from_metadata(self, metadata_dict: Dict[str, Any]) -> Optional[BaseSkill]:
        """Create a SKILL instance from metadata.

        Args:
            metadata_dict: SKILL metadata dictionary

        Returns:
            SKILL instance or None if creation failed
        """
        try:
            # Parse metadata
            metadata = SkillMetadata.from_dict(metadata_dict)

            # Create a simple function-based SKILL
            from ..skills.base_skill import FunctionSkill

            # Create a dummy function
            async def dummy_execute(input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                return {
                    "status": "success",
                    "message": f"Skill '{metadata.name}' executed",
                    "input": input_data
                }

            # Create function-based SKILL
            skill = FunctionSkill(
                metadata=metadata,
                func=dummy_execute,
                auto_validate=True
            )

            self.logger.info(f"Created SKILL from metadata: {metadata.name}")
            return skill

        except Exception as e:
            self.logger.error(f"Failed to create SKILL from metadata: {e}")
            return None