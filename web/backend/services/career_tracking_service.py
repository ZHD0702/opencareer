from __future__ import annotations

import json
import re
from typing import Any

from db.crud import (
    get_pending_skill_follow_up,
    list_job_applications,
    list_skill_evidence_items,
    resolve_skill_follow_up,
    save_job_application,
    save_skill_evidence_item,
    save_skill_follow_up,
    upsert_skill_evidence,
)


STAGE_KEYWORDS = {
    "offered": ("offer", "录用", "拿到意向"),
    "interview": ("面试", "一面", "二面", "终面"),
    "test": ("笔试", "测评", "机考"),
    "applied": ("投递", "投了", "申请了", "已申请"),
    "closed": ("拒绝", "挂了", "没通过", "不考虑了"),
    "saved": ("收藏", "看中", "关注"),
}

SKILL_CATALOG = {
    "Java": "编程语言", "Python": "编程语言", "JavaScript": "编程语言", "TypeScript": "编程语言",
    "C++": "编程语言", "Go": "编程语言", "React": "框架", "Vue": "框架", "Spring Boot": "框架",
    "MySQL": "数据库", "Redis": "数据库", "Docker": "DevOps", "Kubernetes": "DevOps",
    "数据分析": "专业能力", "产品设计": "专业能力", "项目管理": "专业能力", "沟通": "软技能",
    "抗压": "软技能", "协作": "软技能", "领导力": "软技能",
}

FIELD_PRIORITY = ("result", "action", "scenario", "metric", "user_role")
FIELD_QUESTIONS = {
    "scenario": "你当时是在什么项目或工作场景里用到 {skill} 的？",
    "action": "这件事里你具体用 {skill} 做了什么？尽量说说你的实际动作。",
    "result": "做完之后带来了什么变化？没有精确数字，说大致效果也可以。",
    "metric": "这个结果大概能用什么数字或范围来说明？没有精确统计也没关系。",
    "user_role": "这部分主要是你独立负责、协作完成，还是由你主导的？",
}


async def update_career_tracking(
    session_id: str,
    message: str,
    resume_update: dict[str, Any],
    allow_follow_up: bool = True,
) -> dict[str, Any]:
    jobs = _extract_job_update(session_id, message)
    skill_result = await _extract_skill_evidence(session_id, message, resume_update, allow_follow_up)
    return {"jobs_updated": jobs, **skill_result}


def _extract_job_update(session_id: str, message: str) -> int:
    stage = next((key for key, words in STAGE_KEYWORDS.items() if any(word.lower() in message.lower() for word in words)), None)
    if not stage:
        return 0

    existing = list_job_applications(session_id)
    company_match = re.search(r"([A-Za-z0-9\u4e00-\u9fff]{2,16})(?:公司|集团|科技|网络|银行|证券)", message)
    role_match = re.search(r"([A-Za-z0-9+#.\u4e00-\u9fff]{2,20})(?:岗|岗位|工程师|开发|设计师|经理|运营|产品)", message)
    company = company_match.group(0) if company_match else ""
    role = role_match.group(0) if role_match else ""
    candidate = next((card for card in existing if (company and company in card["company"]) or (role and role in card["role"])), None)
    next_action = {
        "applied": "关注进度并准备可能的笔试或面试",
        "test": "完成笔试并整理题目",
        "interview": "准备面试并记录复盘",
        "offered": "核对薪资、职责和入职时间",
    }.get(stage)
    payload = {
        "stage": stage,
        "company": company or (candidate or {}).get("company") or "待确认公司",
        "role": role or (candidate or {}).get("role") or "待确认岗位",
        "next_action": next_action,
        "note": message[:300],
        "source": "conversation",
    }
    if candidate:
        payload["id"] = candidate["id"]
    save_job_application(session_id, payload)
    return 1


async def _extract_skill_evidence(
    session_id: str,
    message: str,
    resume_update: dict[str, Any],
    allow_follow_up: bool,
) -> dict[str, Any]:
    pending = get_pending_skill_follow_up(session_id)
    extracted = await _run_skill_extractor(message, resume_update, pending)
    evidence_items = extracted.get("evidence", []) if extracted else []
    if not evidence_items:
        evidence_items = _fallback_evidence(message, resume_update, pending)

    updated_ids: list[int] = []
    touched_skills: set[str] = set()
    pending_answered = False
    for item in evidence_items:
        skill_name = _normalize_skill_name(item.get("skill_name"))
        if not skill_name:
            continue
        item["skill_name"] = skill_name
        item["source"] = item.get("source") or "conversation"
        item["raw_text"] = item.get("raw_text") or message[:1000]
        if pending and (item.get("id") == pending.get("evidence_id") or skill_name == pending.get("skill_name")):
            item["id"] = pending.get("evidence_id")
            pending_answered = True
        saved = save_skill_evidence_item(session_id, item)
        if saved:
            updated_ids.append(saved["id"])
            touched_skills.add(skill_name)

    if pending_answered and pending:
        resolve_skill_follow_up(session_id, pending["id"])

    for skill_name in touched_skills:
        _refresh_skill_profile(session_id, skill_name)

    follow_up = None
    if allow_follow_up and updated_ids:
        follow_up = _choose_follow_up(session_id, updated_ids, pending if pending_answered else None)

    return {
        "skills_updated": len(touched_skills),
        "evidence_updated": len(updated_ids),
        "follow_up": follow_up,
    }


async def _run_skill_extractor(
    message: str,
    resume_update: dict[str, Any],
    pending: dict[str, Any] | None,
) -> dict[str, Any] | None:
    try:
        from llm.registry import get_llm_adapter

        state = resume_update.get("state") or {}
        context = {
            "known_skills": state.get("skills") or {},
            "latest_experience": (state.get("experiences") or [None])[0],
            "pending_follow_up": pending,
        }
        prompt = f"""
用户最新输入：{message}
已有上下文：{json.dumps(context, ensure_ascii=False)}

抽取用户明确陈述的技能证据。若用户正在回答 pending_follow_up，请更新对应 evidence_id。
只返回 JSON：
{{
  "evidence": [{{
    "id": null,
    "skill_name": "",
    "context_key": "项目、公司或经历的简短稳定标识",
    "scenario": null,
    "task": null,
    "action": null,
    "result": null,
    "metric": null,
    "user_role": null,
    "used_at": null,
    "confidence": 0.0
  }}]
}}
不要编造数字或职责。仅提到技能名称时保留空字段。没有技能信息时返回空数组。
"""
        response = await get_llm_adapter().invoke(
            [{"role": "user", "content": prompt}],
            "你是 OpenCareer 的技能证据抽取器，只返回合法 JSON。",
        )
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            return json.loads(match.group(0)) if match else None
    except Exception:
        return None


def _fallback_evidence(
    message: str,
    resume_update: dict[str, Any],
    pending: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if pending and pending.get("evidence_id"):
        return [{
            "id": pending["evidence_id"],
            "skill_name": pending["skill_name"],
            pending["missing_field"]: message.strip(),
            "confidence": 0.55,
        }]

    state = resume_update.get("state") or {}
    known = ((state.get("skills") or {}).get("hard") or []) + ((state.get("skills") or {}).get("soft") or [])
    found = {skill for skill in SKILL_CATALOG if skill.lower() in message.lower()}
    found.update(str(skill) for skill in known if str(skill).lower() in message.lower())
    has_action = any(word in message for word in ("负责", "使用", "通过", "搭建", "开发", "设计", "优化", "主导", "参与"))
    has_result = any(word in message for word in ("提升", "降低", "减少", "完成", "实现", "节省", "增长"))
    metric = (re.search(r"\d+(?:\.\d+)?\s*(?:%|ms|秒|分钟|小时|天|万|次|人)", message) or [None])[0]
    return [{
        "skill_name": skill,
        "context_key": message[:40],
        "scenario": message if ("项目" in message or "工作" in message or "实习" in message) else None,
        "action": message if has_action else None,
        "result": message if has_result else None,
        "metric": metric,
        "user_role": "主导" if "主导" in message else ("独立负责" if "独立" in message else None),
        "confidence": 0.6,
    } for skill in found]


def _normalize_skill_name(value: Any) -> str | None:
    name = str(value or "").strip()
    if not name or len(name) > 40:
        return None
    for known in SKILL_CATALOG:
        if known.lower() == name.lower():
            return known
    return name


def _evidence_level(item: dict[str, Any]) -> tuple[str, int]:
    required = sum(bool(item.get(field)) for field in ("scenario", "action", "result"))
    optional = sum(bool(item.get(field)) for field in ("metric", "user_role", "used_at"))
    score = required * 25 + optional * 8
    if required == 3 and item.get("metric") and item.get("user_role"):
        return "strong", min(score, 100)
    if required == 3:
        return "proven", min(score, 100)
    if required >= 2:
        return "used", min(score, 100)
    return "mentioned", min(score, 100)


def get_skill_evidence_chains(session_id: str) -> dict[str, list[dict[str, Any]]]:
    chains: dict[str, list[dict[str, Any]]] = {}
    for raw_item in list_skill_evidence_items(session_id):
        item = dict(raw_item)
        item["level"], item["completeness"] = _evidence_level(item)
        item["missing_fields"] = [field for field in FIELD_PRIORITY if not item.get(field)]
        chains.setdefault(item["skill_name"], []).append(item)
    return chains


def _refresh_skill_profile(session_id: str, skill_name: str) -> None:
    items = list_skill_evidence_items(session_id, skill_name)
    if not items:
        return
    ranked = sorted(items, key=lambda item: _evidence_level(item)[1], reverse=True)
    best = ranked[0]
    level, _ = _evidence_level(best)
    status = "proven" if level in {"proven", "strong"} else "mentioned"
    evidence = "；".join(filter(None, (best.get("scenario"), best.get("action"), best.get("result"), best.get("metric"))))
    missing = next((field for field in FIELD_PRIORITY if not best.get(field)), None)
    suggestion = FIELD_QUESTIONS[missing].format(skill=skill_name) if missing else None
    upsert_skill_evidence(session_id, {
        "skill_name": skill_name,
        "status": status,
        "category": SKILL_CATALOG.get(skill_name, "专业能力"),
        "evidence": evidence or best.get("raw_text"),
        "suggestion": suggestion,
        "source": "evidence_chain",
    })


def _choose_follow_up(
    session_id: str,
    evidence_ids: list[int],
    previous: dict[str, Any] | None,
) -> dict[str, Any] | None:
    items = [item for item in list_skill_evidence_items(session_id) if item["id"] in evidence_ids]
    candidates = []
    for item in items:
        missing = next((field for field in FIELD_PRIORITY if not item.get(field)), None)
        if missing:
            candidates.append((FIELD_PRIORITY.index(missing), -float(item.get("confidence") or 0), item, missing))
    if not candidates:
        return None
    _, _, item, missing = sorted(candidates, key=lambda value: (value[0], value[1]))[0]
    asked_count = int((previous or {}).get("asked_count") or 0) + (1 if previous else 0)
    if previous and previous.get("evidence_id") == item["id"] and asked_count >= 2:
        return None
    question = FIELD_QUESTIONS[missing].format(skill=item["skill_name"])
    return save_skill_follow_up(session_id, {
        "evidence_id": item["id"],
        "skill_name": item["skill_name"],
        "missing_field": missing,
        "question": question,
        "asked_count": asked_count,
    })
