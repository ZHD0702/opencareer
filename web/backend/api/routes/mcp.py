from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from adapters.mcp_service import get_mcp_service
from careers_config import config
from services.mcp_resume_service import (
    call_resume_skill as invoke_resume_skill,
    load_mcp_tools,
)

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


async def _load_mcp_tools() -> dict[str, Any]:
    try:
        return await load_mcp_tools()
    except Exception as exc:
        logger.error("Failed to load MCP tools: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Failed to load MCP tools: {exc}") from exc


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
    try:
        result = await invoke_resume_skill(payload.model_dump())
        return {
            "success": True,
            "tool": "resume_skill",
            "result": result,
        }
    except Exception as exc:
        logger.error("resume_skill MCP call failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"resume_skill MCP call failed: {exc}") from exc
