from __future__ import annotations

import re
from typing import Any

from db.crud import get_messages, list_skill_evidence
from services.resume_builder_service import ResumeBuilderService


SEARCH_INTENT_PATTERNS = (
    r"帮我找(?:一下)?(?:工作|岗位|职位)",
    r"(?:找|搜|看看|推荐)(?:一下)?(?:合适的)?(?:工作|岗位|职位)",
    r"有哪些.*(?:岗位|职位|工作)",
    r"有没有.*(?:岗位|职位|工作)",
    r"想找.*(?:工作|岗位|职位)",
    r"匹配岗位",
)

CITY_CODES = {
    "全国": "489", "北京": "530", "上海": "538", "广州": "763", "深圳": "765",
    "杭州": "653", "成都": "801", "南京": "635", "武汉": "736", "西安": "854",
    "苏州": "639", "天津": "531", "重庆": "551", "厦门": "682", "长沙": "749",
    "郑州": "719", "青岛": "702",
}


def evaluate_job_search_readiness(session_id: str) -> dict[str, Any]:
    state = ResumeBuilderService().load_state(session_id)
    target = state.get("target") or {}
    basics = state.get("basics") or {}
    skills = list_skill_evidence(session_id)
    messages = get_messages(session_id, limit=100)
    user_text = "\n".join(item["content"] for item in messages if item["role"] == "user")

    role = (target.get("role") or "").strip()
    city = (target.get("city") or "").strip()
    background_known = bool(
        basics.get("grade_level") or basics.get("major") or basics.get("school")
        or state.get("education") or state.get("experiences")
    )
    proven = [item for item in skills if item.get("status") == "proven"]
    known_skills = [item for item in skills if item.get("status") in {"proven", "mentioned"}]
    requested = any(re.search(pattern, user_text, re.IGNORECASE) for pattern in SEARCH_INTENT_PATTERNS)

    missing = []
    if not role:
        missing.append("target_role")
    if not city:
        missing.append("city")
    if not background_known:
        missing.append("background")
    if len(known_skills) < 2:
        missing.append("skills")
    if not proven:
        missing.append("proven_evidence")

    ready = not missing
    top_skills = [item["skill_name"] for item in proven[:5]]
    employment_type = "实习" if "实习" in user_text or "实习" in role else ""
    return {
        "ready": ready,
        "requested": requested,
        "missing_fields": missing,
        "query_plan": {
            "role": role,
            "city": city,
            "city_code": CITY_CODES.get(city, CITY_CODES["全国"]),
            "salary": target.get("salary_expectation"),
            "skills": top_skills,
            "employment_type": employment_type,
            "major": basics.get("major"),
            "industry": target.get("industry"),
        } if ready else None,
    }


def build_job_match_action(session_id: str, allow_action: bool = True) -> dict[str, Any] | None:
    if not allow_action:
        return None
    assessment = evaluate_job_search_readiness(session_id)
    if not assessment["ready"] or not assessment["requested"]:
        return None
    existing_messages = get_messages(session_id, limit=100)
    if any(
        action.get("action") == "start_job_matching"
        for message in existing_messages
        for action in (message.get("actions") or [])
    ):
        return None
    return {
        "action": "start_job_matching",
        "label": "匹配岗位",
        "query_plan": assessment["query_plan"],
    }
