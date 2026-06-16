from __future__ import annotations

import re
from typing import Any

from db.crud import get_messages, list_skill_evidence
from services.resume_builder_service import ResumeBuilderService


SEARCH_INTENT_PATTERNS = (
    r"帮我找.*(?:工作|岗位|职位|实习)",
    r"(?:找|搜|看看|推荐|匹配).*(?:工作|岗位|职位|实习)",
    r"有哪些.*(?:岗位|职位|工作|实习)",
    r"有没有.*(?:岗位|职位|工作|实习)",
    r"想找.*(?:工作|岗位|职位|实习)",
    r"匹配岗位",
)

CITY_CODES = {
    "全国": "489", "北京": "530", "上海": "538", "广州": "763", "深圳": "765",
    "杭州": "653", "成都": "801", "南京": "635", "武汉": "736", "西安": "854",
    "苏州": "639", "天津": "531", "重庆": "551", "厦门": "682", "长沙": "749",
    "郑州": "719", "青岛": "702",
}

INTERNSHIP_INTENT_MARKERS = ("实习", "实习生", "暑期实习", "日常实习", "intern", "internship")
FULLTIME_INTENT_MARKERS = ("全职", "正式", "社招", "校招", "应届生", "毕业生", "fulltime", "full-time")


def _infer_employment_type(text: str) -> str:
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in INTERNSHIP_INTENT_MARKERS):
        return "实习"
    if any(marker.lower() in lowered for marker in FULLTIME_INTENT_MARKERS):
        return "全职"
    return ""


def evaluate_job_search_readiness(session_id: str) -> dict[str, Any]:
    state = ResumeBuilderService().load_state(session_id)
    target = state.get("target") or {}
    basics = state.get("basics") or {}
    skills = list_skill_evidence(session_id)
    messages = get_messages(session_id, limit=100)
    user_messages = [item["content"] for item in messages if item["role"] == "user"]
    user_text = "\n".join(user_messages)
    latest_user_text = user_messages[0] if user_messages else ""

    role = (target.get("role") or "").strip()
    city = (target.get("city") or "").strip()
    background_known = bool(
        basics.get("grade_level") or basics.get("major") or basics.get("school")
        or state.get("education") or state.get("experiences")
    )
    proven = [item for item in skills if item.get("status") == "proven"]
    known_skills = [item for item in skills if item.get("status") in {"proven", "mentioned"}]
    resume_skills = [
        str(item).strip()
        for item in (state.get("skills") or {}).get("hard") or []
        if str(item).strip()
    ]
    requested = any(re.search(pattern, latest_user_text, re.IGNORECASE) for pattern in SEARCH_INTENT_PATTERNS)

    missing = []
    if not role:
        missing.append("target_role")
    if not background_known:
        missing.append("background")
    if not known_skills and not resume_skills:
        missing.append("skills")

    ready = not missing
    top_skills = list(dict.fromkeys(
        [item["skill_name"] for item in proven]
        + [item["skill_name"] for item in known_skills]
        + resume_skills
    ))[:5]
    search_city = city or "全国"
    employment_type = _infer_employment_type(f"{role}\n{user_text}")
    return {
        "ready": ready,
        "requested": requested,
        "missing_fields": missing,
        "query_plan": {
            "role": role,
            "city": search_city,
            "city_code": CITY_CODES.get(search_city, CITY_CODES["全国"]),
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
    return {
        "action": "start_job_matching",
        "label": "匹配岗位",
        "query_plan": assessment["query_plan"],
    }
