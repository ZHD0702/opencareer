"""Prompt/response template loader and skill registry for MCP server tools."""

import logging
import re
from inspect import iscoroutinefunction
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


def _build_skill_output(result: dict) -> str:
    """Build a human-readable output summary from a tool result dict."""
    parts: list[str] = []
    if result.get("pdf_path"):
        parts.append(f"PDF 已生成：{result['pdf_path']}")
    if result.get("json_path"):
        parts.append(f"JSON 已保存：{result['json_path']}")
    if result.get("ats_report"):
        ats = result["ats_report"]
        parts.append(f"ATS 评分：{ats.get('score', 'N/A')}")
        suggestions = ats.get("suggestions", [])
        if suggestions:
            parts.append("优化建议：" + "；".join(suggestions))
    if result.get("notes"):
        parts.extend(n for n in result["notes"] if n)
    return "\n".join(parts) if parts else "操作完成。"


def _parse_skill_md(md_file: Path) -> Dict[str, Any]:
    """Parse a SKILL.md file and return a skill definition dict."""
    text = md_file.read_text(encoding="utf-8")
    
    def _extract_section(header: str) -> str:
        pattern = rf"^##\s*{re.escape(header)}\s*\n(.*?)(?=^##\s|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        return match.group(1).strip() if match else ""
    
    title_match = re.match(r"^#\s*(.+)", text, re.MULTILINE)
    skill_id = title_match.group(1).strip() if title_match else md_file.parent.name
    
    triggers_text = _extract_section("Triggers")
    triggers = [t.strip().lstrip("- ").strip() for t in triggers_text.split("\n") if t.strip()]
    
    tool_ref = _extract_section("Tool")
    return {
        "skill_id": skill_id,
        "name": _extract_section("Description") or skill_id,
        "description": _extract_section("Description"),
        "triggers": triggers,
        "instructions": _extract_section("Instructions"),
        "requirements": _extract_section("Requirements"),
        "tool": tool_ref if tool_ref else None,
        "source": "markdown",
    }


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
        """Load skill definitions from skills.yaml and SKILL.md files."""
        skills = {}

        try:
            data = _load_yaml("skills")
            skills.update(data.get("skills", {}))
        except Exception as e:
            logger.warning("Failed to load skills.yaml: %s", e)

        skills_dir = _PROMPTS_DIR.parent / "skills"
        if skills_dir.exists():
            for skill_dir in sorted(skills_dir.iterdir()):
                if skill_dir.is_dir():
                    md_file = skill_dir / "SKILL.md"
                    if md_file.exists():
                        try:
                            skill_def = _parse_skill_md(md_file)
                            skills[skill_def["skill_id"]] = skill_def
                            logger.info("Loaded SKILL.md from: %s", skill_dir.name)
                        except Exception as e:
                            logger.warning("Failed to parse %s: %s", md_file, e)

        return skills

    def register_all(self):
        """Register all skills as MCP tools."""
        for skill_id, skill_def in self.skills.items():
            self._register_skill(skill_id, skill_def)

    def _register_skill(self, skill_id: str, skill_def: Dict[str, Any]):
        """Convert a single skill definition to an MCP tool."""
        name = skill_def.get("name", skill_id)
        description = skill_def.get("description", name)
        source = skill_def.get("source", "yaml")

        if source == "markdown":
            instructions = skill_def.get("instructions", "")
            tool_name = skill_def.get("tool")

            async def md_executor(**kwargs):
                """Execute SKILL.md by following its instructions.
                If a tool is bound, forwards kwargs to the registered tool.
                Otherwise returns instructions as structured output.
                """
                if tool_name and tool_name in self.tools:
                    try:
                        fn = self.tools[tool_name]
                        if iscoroutinefunction(fn):
                            result = await fn(**kwargs)
                        else:
                            result = fn(**kwargs)
                        result["_skill"] = skill_id
                        result["_type"] = "markdown_skill"
                        result["output"] = _build_skill_output(result)
                        return result
                    except Exception as exc:
                        return {
                            "_skill": skill_id,
                            "_type": "markdown_skill",
                            "ok": False,
                            "error": f"Tool '{tool_name}' failed: {exc}",
                            "instructions": instructions,
                            "output": f"执行失败：{exc}",
                        }
                return {
                    "_skill": skill_id,
                    "_type": "markdown_skill",
                    "instructions": instructions,
                    "output": instructions,
                }

            executor = md_executor
        else:
            parameters = skill_def.get("parameters", {})
            steps = skill_def.get("steps", [])

            async def yaml_executor(**kwargs):
                """Execute skill by running steps in order."""
                results = {}
                for i, step in enumerate(steps):
                    tool_name = step["tool"]
                    if tool_name not in self.tools:
                        results[f"step_{i}_error"] = f"Tool '{tool_name}' not found"
                        continue
                    args = self._resolve_args(step["args"], kwargs)
                    try:
                        fn = self.tools[tool_name]
                        if iscoroutinefunction(fn):
                            results[f"step_{i}_{tool_name}"] = await fn(**args)
                        else:
                            results[f"step_{i}_{tool_name}"] = fn(**args)
                    except Exception as e:
                        results[f"step_{i}_error"] = str(e)
                return results

            executor = yaml_executor

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
