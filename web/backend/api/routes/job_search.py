from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from db.crud import get_session
from services.job_search_readiness import evaluate_job_search_readiness
from services.zhaopin_job_service import ZhaopinJobService, ZhaopinSearchError


router = APIRouter()


class JobMatchRequest(BaseModel):
    session_id: str
    role: str | None = Field(default=None, max_length=80)
    city: str | None = Field(default=None, max_length=20)
    city_code: str | None = Field(default=None, pattern=r"^\d{2,5}$")
    salary: str | None = None
    skills: list[str] = Field(default_factory=list)
    employment_type: str | None = None
    major: str | None = None
    industry: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


@router.post("/job-search/match")
async def match_jobs(payload: JobMatchRequest):
    if not get_session(payload.session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    assessment = evaluate_job_search_readiness(payload.session_id)
    plan = assessment.get("query_plan") or {}
    overrides = payload.model_dump(exclude={"session_id", "limit"}, exclude_none=True)
    for key, value in overrides.items():
        if value not in (None, "", []):
            plan[key] = value
    if not plan.get("role"):
        raise HTTPException(status_code=409, detail="请先确认目标岗位")

    try:
        result = await ZhaopinJobService().search_and_match(plan, payload.limit)
    except ZhaopinSearchError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": str(exc),
                "code": "zhaopin_search_failed",
                "query_plan": plan,
            },
        ) from exc
    return {"session_id": payload.session_id, "query_plan": plan, **result}
