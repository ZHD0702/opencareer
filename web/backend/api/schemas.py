from pydantic import BaseModel, Field
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
    current_overall_state: Optional[str] = "neutral"
    current_emotions: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    support_intensity: str = "none"
    suggested_action: str = "work"
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
    grade_level: Optional[str] = None
    major: Optional[str] = None
    school: Optional[str] = None
    target_role: Optional[str] = None
    job_search_stage: Optional[str] = None
    skill_focus: List[str] = Field(default_factory=list)
    common_concerns: List[str] = Field(default_factory=list)
    background_summary: Optional[str] = None
    resume_state: Optional[dict] = None


class ResumeResponse(BaseModel):
    session_id: str
    data: ResumeData
    last_updated: str
