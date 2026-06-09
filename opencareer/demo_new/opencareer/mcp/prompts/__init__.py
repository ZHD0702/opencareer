"""Prompt/response template loader and skill registry for MCP server tools."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict

import yaml

logger = logging.getLogger("opencareer.mcp.prompts")

_PROMPTS_DIR = Path(__file__).parent

_cache: Dict[str, Any] = {}


def _load_yaml(name: str) -> Any:
    """Load a YAML file from prompts/ directory with simple caching."""
    if name in _cache:
        return _cache[name]
    path = _PROMPTS_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    _cache[name] = data
    return data


def get_emotion_templates() -> tuple[Dict[str, Dict[str, str]], list[str]]:
    """Return (emotion_responses_dict, support_tips_list)."""
    data = _load_yaml("emotion_support")
    return data["responses"], data["support_tips"]


def get_career_templates() -> tuple[Dict[str, Dict[str, Any]], str]:
    """Return (topics_dict, next_step_suggestion)."""
    data = _load_yaml("career_advice")
    return data["topics"], data["next_step_suggestion"]


def get_interview_templates() -> tuple[Dict[str, list[str]], list[str], str]:
    """Return (questions_dict, tips_list, star_reminder)."""
    data = _load_yaml("interview_prep")
    return data["questions"], data["tips"], data["star_reminder"]


# ------------------------------------------------------------------
# Skill Registry: auto-converts Skill definitions to MCP Tools
# ------------------------------------------------------------------


class SkillRegistry:
    """Loads skill definitions from YAML and creates tool executors."""

    def __init__(self, mcp_app, tool_catalog: Dict[str, Callable]):
        """
        Args:
            mcp_app: FastMCP instance to register tools on.
            tool_catalog: dict mapping tool_name -> async_function.
                          These are the base tools that skills can call.
        """
        self.mcp = mcp_app
        self.tools = tool_catalog
        self.skills = self._load_skills()

    def _load_skills(self) -> Dict[str, Any]:
        """Load skill definitions from skills.yaml."""
        try:
            data = _load_yaml("skills")
            return data.get("skills", {})
        except Exception as e:
            logger.warning("Failed to load skills: %s", e)
            return {}

    def register_all(self):
        """Register all skills as MCP tools."""
        for skill_id, skill_def in self.skills.items():
            self._register_skill(skill_id, skill_def)

    def _register_skill(self, skill_id: str, skill_def: Dict[str, Any]):
        """Convert a single skill definition to an MCP tool."""
        name = skill_def["name"]
        description = skill_def["description"]
        parameters = skill_def.get("parameters", {})
        steps = skill_def.get("steps", [])

        async def executor(**kwargs):
            """Execute skill by running steps in order."""
            results = {}
            for i, step in enumerate(steps):
                tool_name = step["tool"]
                if tool_name not in self.tools:
                    results[f"step_{i}_error"] = f"Tool '{tool_name}' not found"
                    continue
                args = self._resolve_args(step["args"], kwargs)
                try:
                    results[f"step_{i}_{tool_name}"] = await self.tools[tool_name](**args)
                except Exception as e:
                    results[f"step_{i}_error"] = str(e)
            return results

        executor.__name__ = skill_id
        executor.__doc__ = description

        self.mcp.tool(name=skill_id, description=description)(executor)

        logger.info("Registered skill as tool: %s (%s)", skill_id, name)

    def _resolve_args(self, arg_template: Dict[str, str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve {param_name} placeholders in step args with actual values."""
        resolved = {}
        for key, value in arg_template.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                param_name = value[1:-1]
                resolved[key] = kwargs.get(param_name)
            else:
                resolved[key] = value
        return resolved


def create_skill_registry(mcp_app, tool_catalog: Dict[str, Callable]) -> SkillRegistry:
    """Create and register all skills as MCP tools."""
    registry = SkillRegistry(mcp_app, tool_catalog)
    registry.register_all()
    return registry
