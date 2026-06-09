"""Prompt/response template loader and skill registry for MCP server tools."""

import logging
import re
from inspect import iscoroutinefunction
from pathlib import Path
from typing import Any, Callable, Dict

import yaml

logger = logging.getLogger("careers.mcp.prompts")

_PROMPTS_DIR = Path(__file__).parent

_cache: Dict[str, Any] = {}


def _load_yaml(name: str) -> Any:
    """Load a YAML file from prompts/ directory with simple caching."""
    if name in _cache:
        return _cache[name]
    path = _PROMPTS_DIR / f"{name}.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _cache[name] = data
        return data
    return {}


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


def create_skill_registry(mcp_app, tool_catalog: Dict[str, Callable]):
    """Placeholder for future skill registry."""
    return None
