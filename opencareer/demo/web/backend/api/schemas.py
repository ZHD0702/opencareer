from pydantic import BaseModel
from typing import Optional, List


class CreateSessionRequest(BaseModel):
    user_id: str
    target_role: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    target_role: Optional[str]
    created_at: str


class ChatRequest(BaseModel):
    message: str


class EmotionTrendsResponse(BaseModel):
    session_id: str
    current_mood: Optional[str]
    trend: str
    consecutive_negative: int
    negative_ratio: float
    needs_intervention: bool
    reason: str
    history: List[dict]


class SkillItem(BaseModel):
    name: str
    level: int
    required: int
    category: str


class GapItem(BaseModel):
    skill: str
    gap: int
    suggestion: str


class SkillAssessmentResponse(BaseModel):
    session_id: str
    target_role: str
    match_rate: int
    skills: List[SkillItem]
    gaps: List[GapItem]


class ResumeData(BaseModel):
    grade_level: Optional[str]
    major: Optional[str]
    school: Optional[str]
    target_role: Optional[str]
    job_search_stage: Optional[str]
    skill_focus: List[str]
    common_concerns: List[str]
    background_summary: Optional[str]


class ResumeResponse(BaseModel):
    session_id: str
    data: ResumeData
    last_updated: str
