from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from db.crud import get_session
from services.browser_session_service import browser_session_service


router = APIRouter()


class BrowserSessionRequest(BaseModel):
    session_id: str
    role: str = Field(min_length=1, max_length=80)
    city: str = Field(default="全国", max_length=20)
    city_code: str = Field(default="489", pattern=r"^\d{2,5}$")
    salary: str | None = None
    skills: list[str] = Field(default_factory=list)
    employment_type: str | None = None
    major: str | None = None
    industry: str | None = None


class BrowserInteractionRequest(BaseModel):
    type: str
    x: float | None = None
    y: float | None = None
    delta_y: float | None = None
    text: str | None = None
    key: str | None = None


@router.post("/browser-sessions")
async def create_browser_session(payload: BrowserSessionRequest):
    if not get_session(payload.session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return await browser_session_service.create(payload.session_id, payload.model_dump(exclude={"session_id"}))


@router.get("/browser-sessions/{task_id}")
async def get_browser_session(task_id: str):
    task = browser_session_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="浏览器会话不存在")
    return task.public()


@router.get("/browser-sessions/{task_id}/screenshot")
async def get_browser_screenshot(task_id: str):
    try:
        content = await browser_session_service.screenshot(task_id)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not content:
        raise HTTPException(status_code=409, detail="浏览器画面尚未就绪")
    return Response(content=content, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.post("/browser-sessions/{task_id}/interact")
async def interact_with_browser(task_id: str, payload: BrowserInteractionRequest):
    try:
        return await browser_session_service.interact(task_id, payload.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/browser-sessions/{task_id}/resume")
async def resume_browser_session(task_id: str):
    try:
        return await browser_session_service.resume(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/browser-sessions/{task_id}")
async def stop_browser_session(task_id: str):
    return {"stopped": await browser_session_service.stop(task_id)}
