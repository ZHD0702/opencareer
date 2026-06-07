from __future__ import annotations

import logging
import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from adapters.mcp_service import get_mcp_service, start_mcp_server
from careers_config import config

router = APIRouter()
logger = logging.getLogger(__name__)


class ResumeSkillRequest(BaseModel):
    action: Literal["generate", "optimize", "ats_check", "export_pdf"] = "generate"
    name: str = ""
    target_role: str = ""
    job_type: str = ""
    industry: str = ""
    city: str = ""
    experience_years: str = ""
    phone: str = ""
    email: str = ""
    education: str = ""
    summary: str = ""
    experiences: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    jd: str = ""
    bilingual: bool = False
    output_json_path: str = ""
    output_pdf_path: str = ""


async def _ensure_mcp_available() -> None:
    if not config.USE_MCP:
        raise HTTPException(status_code=503, detail="MCP is disabled. Set CAREER_USE_MCP=true.")

    started = await start_mcp_server()
    if not started:
        raise HTTPException(status_code=503, detail="MCP server is not available.")


async def _load_mcp_tools() -> dict[str, Any]:
    await _ensure_mcp_available()

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {
                "opencareer": {
                    "transport": "streamable_http",
                    "url": config.MCP_URL,
                }
            }
        )
        tools = await client.get_tools()
        return {tool.name: tool for tool in tools}
    except Exception as exc:
        logger.error("Failed to load MCP tools: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Failed to load MCP tools: {exc}") from exc


def _normalize_tool_result(result: Any) -> Any:
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        text = result[0].get("text")
        if isinstance(text, str):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return result


@router.get("/mcp/status")
async def get_mcp_status():
    service = await get_mcp_service()
    tools: list[str] = []
    error = None

    if config.USE_MCP:
        try:
            tools = sorted((await _load_mcp_tools()).keys())
        except HTTPException as exc:
            error = exc.detail

    return {
        "enabled": config.USE_MCP,
        "running": service.is_running(),
        "url": config.MCP_URL if config.USE_MCP else None,
        "tools": tools,
        "resume_skill_available": "resume_skill" in tools,
        "error": error,
    }


@router.get("/mcp/tools")
async def list_mcp_tools():
    tools = await _load_mcp_tools()
    return {
        "success": True,
        "tools": sorted(tools.keys()),
    }


@router.post("/mcp/resume-skill")
async def call_resume_skill(payload: ResumeSkillRequest):
    tools = await _load_mcp_tools()
    tool = tools.get("resume_skill")
    if tool is None:
        raise HTTPException(status_code=404, detail="resume_skill was not exposed by MCP.")

    try:
        result = await tool.ainvoke(payload.model_dump())
        return {
            "success": True,
            "tool": "resume_skill",
            "result": _normalize_tool_result(result),
        }
    except Exception as exc:
        logger.error("resume_skill MCP call failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"resume_skill MCP call failed: {exc}") from exc
