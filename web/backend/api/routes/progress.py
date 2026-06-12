from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from db.crud import delete_job_application, get_session, list_job_applications, save_job_application, upsert_skill_evidence
from services.career_tracking_service import SKILL_CATALOG


router = APIRouter()
STAGES = ["saved", "applied", "test", "interview", "offered", "closed"]


class JobApplicationPayload(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    stage: Optional[str] = None
    next_action: Optional[str] = None
    deadline: Optional[str] = None
    jd_text: Optional[str] = None
    note: Optional[str] = None
    to_stage: Optional[str] = None


def _serialize(card: dict) -> dict:
    return {**card, "id": str(card["id"]), "date": card.get("updated_at", "")[:10]}


def _sync_jd_gaps(session_id: str, jd_text: Optional[str]) -> None:
    if not jd_text:
        return
    for skill, category in SKILL_CATALOG.items():
        if skill.lower() in jd_text.lower():
            upsert_skill_evidence(session_id, {
                "skill_name": skill,
                "status": "gap",
                "category": category,
                "requirement": f"目标岗位 JD 提到 {skill}",
                "suggestion": f"补充一段能够证明 {skill} 的项目或工作成果；如果尚未掌握，建立一个可展示的小项目。",
                "source": "job_description",
            })


@router.get("/job-progress/{session_id}")
async def get_job_progress(session_id: str):
    if not get_session(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    cards = [_serialize(card) for card in list_job_applications(session_id)]
    return {"session_id": session_id, "stages": {stage: [c for c in cards if c["stage"] == stage] for stage in STAGES}}


@router.post("/job-progress/{session_id}")
async def create_job_progress(session_id: str, payload: JobApplicationPayload):
    if not payload.company or not payload.role:
        raise HTTPException(status_code=400, detail="公司和岗位不能为空")
    data = payload.model_dump(exclude_none=True)
    data["stage"] = data.get("stage", "saved")
    card = save_job_application(session_id, data)
    _sync_jd_gaps(session_id, data.get("jd_text"))
    return _serialize(card)


@router.patch("/job-progress/{session_id}/{card_id}")
async def update_job_progress(session_id: str, card_id: int, payload: JobApplicationPayload):
    data = payload.model_dump(exclude_none=True)
    if "to_stage" in data:
        data["stage"] = data.pop("to_stage")
    data["id"] = card_id
    card = save_job_application(session_id, data)
    if not card:
        raise HTTPException(status_code=404, detail="岗位记录不存在")
    _sync_jd_gaps(session_id, data.get("jd_text"))
    return _serialize(card)


@router.delete("/job-progress/{session_id}/{card_id}")
async def remove_job_progress(session_id: str, card_id: int):
    return {"deleted": delete_job_application(session_id, card_id)}
