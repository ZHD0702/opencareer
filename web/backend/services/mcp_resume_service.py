from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from adapters.mcp_service import start_mcp_server
from careers_config import config


logger = logging.getLogger(__name__)

REQUIRED_RESUME_FIELDS = (
    ("name", "姓名", "先告诉我你的姓名吧。"),
    ("phone", "手机号", "再给我一个简历上使用的手机号。"),
    ("email", "邮箱", "简历上准备放哪个邮箱？"),
    ("target_role", "目标岗位", "你这份简历主要准备投什么岗位？"),
    ("education", "教育背景", "请告诉我学校、专业，以及学历或当前年级。"),
    ("experience", "项目或经历", "再讲一段最能代表你的项目、实习或工作经历吧。"),
    ("skills", "技能", "你希望简历重点展示哪些技能？"),
)


def _mcp_url() -> str:
    return config.MCP_URL.replace("http://localhost:", "http://127.0.0.1:")


def _local_http_client_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout,
        auth=auth,
        trust_env=False,
    )


async def load_mcp_tools() -> dict[str, Any]:
    if not config.USE_MCP:
        raise RuntimeError("MCP is disabled. Set CAREER_USE_MCP=true.")
    if not await start_mcp_server():
        raise RuntimeError("MCP server is not available.")

    from langchain_mcp_adapters.client import MultiServerMCPClient

    last_error: BaseException | None = None
    for attempt in range(2):
        try:
            client = MultiServerMCPClient(
                {
                    "opencareer": {
                        "transport": "streamable_http",
                        "url": _mcp_url(),
                        "httpx_client_factory": _local_http_client_factory,
                        "terminate_on_close": False,
                    }
                }
            )
            tools = await asyncio.wait_for(client.get_tools(), timeout=12)
            return {tool.name: tool for tool in tools}
        except BaseException as exc:
            last_error = exc
            logger.warning("MCP tool loading attempt %s failed: %s", attempt + 1, exc)
            if attempt == 0:
                await start_mcp_server()
                await asyncio.sleep(0.4)

    detail = _format_exception(last_error)
    raise RuntimeError(f"Failed to load MCP tools: {detail}") from last_error


def _format_exception(exc: BaseException | None) -> str:
    if exc is None:
        return "unknown error"
    nested = getattr(exc, "exceptions", None)
    if nested:
        messages = [_format_exception(item) for item in nested]
        return "; ".join(message for message in messages if message)
    return f"{type(exc).__name__}: {exc}"


def normalize_tool_result(result: Any) -> Any:
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        text = result[0].get("text")
        if isinstance(text, str):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return result


async def call_resume_skill(payload: dict[str, Any]) -> dict[str, Any]:
    tools = await load_mcp_tools()
    tool = tools.get("resume_skill")
    if tool is None:
        raise RuntimeError("resume_skill was not exposed by MCP.")
    result = normalize_tool_result(await tool.ainvoke(payload))
    if not isinstance(result, dict):
        raise RuntimeError(f"resume_skill returned an invalid result: {result}")
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "resume_skill failed without an error message.")
    return result


def is_resume_pdf_request(text: str, previous_assistant_text: str = "") -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    asks_for_resume = "简历" in compact
    asks_to_generate = any(word in compact for word in ("生成", "导出", "制作", "做一份", "帮我写"))
    asks_for_pdf = "pdf" in compact or "文件" in compact
    if asks_for_resume and asks_to_generate and asks_for_pdf:
        return True

    previous = re.sub(r"\s+", "", previous_assistant_text or "").lower()
    assistant_offered_pdf = (
        "简历" in previous
        and ("pdf" in previous or "文件" in previous)
        and any(word in previous for word in ("生成", "导出", "制作"))
    )
    confirmation = compact.rstrip("。！!，,") in {
        "可以", "可以的", "好", "好的", "行", "需要", "要", "确认", "没问题",
    } or any(phrase in compact for phrase in ("生成吧", "导出吧", "就这样生成", "直接生成"))
    return assistant_offered_pdf and confirmation


def build_resume_skill_payload(
    state: dict[str, Any],
    skill_chains: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    basics = state.get("basics") or {}
    target = state.get("target") or {}
    skills = state.get("skills") or {}

    education_parts = [
        basics.get("school"),
        basics.get("major"),
        basics.get("grade_level"),
    ]
    education = " / ".join(str(item) for item in education_parts if item)

    experiences: list[str] = []
    projects: list[str] = []
    for item in state.get("experiences") or []:
        bullets = [str(value) for value in item.get("bullets") or [] if value]
        values = bullets or ([str(item.get("raw"))] if item.get("raw") else [])
        if item.get("type") == "project":
            projects.extend(values)
        else:
            experiences.extend(values)

    for item in state.get("projects") or []:
        if isinstance(item, str):
            projects.append(item)
        elif isinstance(item, dict):
            projects.extend(str(value) for value in item.get("bullets") or [] if value)
            if item.get("raw"):
                projects.append(str(item["raw"]))

    industry = target.get("industry") or ""
    industry_labels = {
        "internet": "互联网",
        "manufacturing": "制造业",
        "fresh_graduate": "应届生",
    }

    skill_names = _unique(list(skills.get("hard") or []) + list(skills.get("soft") or []))

    return {
        "action": "export_pdf",
        "name": basics.get("name") or "",
        "phone": basics.get("phone") or "",
        "email": basics.get("email") or "",
        "target_role": target.get("role") or "",
        "job_type": basics.get("grade_level") or "",
        "industry": industry_labels.get(industry, industry),
        "city": target.get("city") or "",
        "experience_years": basics.get("grade_level") or "",
        "education": education,
        "summary": state.get("last_update_summary") or "",
        "experiences": _unique(experiences),
        "projects": _unique(projects),
        "skills": [
            f"{skill}（{_skill_proficiency(skill, skill_chains or {})}）"
            for skill in skill_names
        ],
    }


def validate_resume_payload(payload: dict[str, Any]) -> dict[str, Any]:
    education = str(payload.get("education") or "").strip()
    experience_items = list(payload.get("experiences") or []) + list(payload.get("projects") or [])
    values = {
        "name": str(payload.get("name") or "").strip(),
        "phone": str(payload.get("phone") or "").strip(),
        "email": str(payload.get("email") or "").strip(),
        "target_role": str(payload.get("target_role") or "").strip(),
        "education": education if len([part for part in education.split(" / ") if part.strip()]) >= 3 else "",
        "experience": [item for item in experience_items if str(item).strip()],
        "skills": [item for item in payload.get("skills") or [] if str(item).strip()],
    }
    missing = [
        {"key": key, "label": label, "question": question}
        for key, label, question in REQUIRED_RESUME_FIELDS
        if not values[key]
    ]
    return {
        "complete": not missing,
        "missing": missing,
        "missing_labels": [item["label"] for item in missing],
        "next_question": missing[0]["question"] if missing else "",
    }


def build_incomplete_resume_response(validation: dict[str, Any]) -> str:
    labels = validation.get("missing_labels") or []
    missing_text = "、".join(labels)
    question = validation.get("next_question") or "我们先把缺少的信息补完整。"
    return f"现在还不能直接生成，简历里还缺：{missing_text}。这些会直接影响简历能不能正常投递。\n\n{question}"


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item and item.strip()))


def _skill_proficiency(
    skill_name: str,
    skill_chains: dict[str, list[dict[str, Any]]],
) -> str:
    evidence = skill_chains.get(skill_name) or []
    levels = {str(item.get("level") or "") for item in evidence}
    if levels.intersection({"strong", "proven"}):
        return "熟练"
    if "used" in levels or any(
        item.get("action") and (item.get("scenario") or item.get("result"))
        for item in evidence
    ):
        return "掌握"
    return "了解"
