"""
Pydantic models for the OpenCareer Web API.

Defines schemas for REST request/response bodies and WebSocket messages.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Session
# ------------------------------------------------------------------


class SessionCreateRequest(BaseModel):
    """Request to create a new conversation session."""
    user_id: str = "default_user"


class SessionCreateResponse(BaseModel):
    """Response after creating a new session."""
    session_id: str
    user_id: str
    created_at: str


class SessionInfo(BaseModel):
    """Basic session metadata."""
    session_id: str
    user_id: str
    title: str = "新会话"
    preview: str = ""
    turn_count: int = 0
    current_agent: Optional[str] = None
    created_at: str = ""


# ------------------------------------------------------------------
# Chat (REST alternative to WebSocket)
# ------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Non-streaming chat request (fallback)."""
    session_id: str
    content: str


class ChatResponse(BaseModel):
    """Non-streaming chat response."""
    session_id: str
    response: str
    demand_type: str
    agent_used: str
    status: str


# ------------------------------------------------------------------
# Emotion trends
# ------------------------------------------------------------------


class EmotionTrendItem(BaseModel):
    """Single emotion analysis entry."""
    emotions: List[str] = []
    overall_state: str = "neutral"
    support_intensity: str = "none"
    demand_type: str = "unknown"
    confidence: float = 0.0
    timestamp: str = ""


class EmotionTrendsResponse(BaseModel):
    """Emotion trends data for the sidebar."""
    session_id: str
    current_mood: Optional[str] = None
    trend: str = "insufficient_data"  # improving / declining / stable / insufficient_data
    consecutive_negative: int = 0
    negative_ratio: float = 0.0
    needs_intervention: bool = False
    reason: str = ""
    history: List[EmotionTrendItem] = []


# ------------------------------------------------------------------
# Resume / Profile
# ------------------------------------------------------------------


class ResumeData(BaseModel):
    """User profile/resume data extracted from conversation."""
    grade_level: Optional[str] = None
    major: Optional[str] = None
    school: Optional[str] = None
    target_role: Optional[str] = None
    job_search_stage: Optional[str] = None
    skill_focus: List[str] = []
    common_concerns: List[str] = []
    background_summary: Optional[str] = None


class ResumeResponse(BaseModel):
    """Full resume/profile response for GET."""
    session_id: str
    data: ResumeData
    last_updated: str = ""


class ResumeUpdateRequest(BaseModel):
    """Partial update payload for PATCH."""
    grade_level: Optional[str] = None
    major: Optional[str] = None
    school: Optional[str] = None
    target_role: Optional[str] = None
    job_search_stage: Optional[str] = None
    skill_focus: Optional[List[str]] = None
    common_concerns: Optional[List[str]] = None
    background_summary: Optional[str] = None


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    mcp_server: str = "http://localhost:8000"


# ------------------------------------------------------------------
# Job Progress
# ------------------------------------------------------------------


class JobCard(BaseModel):
    """A single job application card."""
    id: str
    company: str
    role: str
    date: str = ""
    note: Optional[str] = None


class JobProgressResponse(BaseModel):
    """Full job progress data for a session."""
    session_id: str
    stages: Dict[str, List[JobCard]]  # stage name -> cards


class JobCardCreateRequest(BaseModel):
    """Payload for creating a new job card."""
    stage: str  # applied / interviewing / offered / rejected
    company: str
    role: str
    date: str = ""
    note: Optional[str] = None


class JobCardMoveRequest(BaseModel):
    """Payload for moving a card to a different stage."""
    to_stage: str  # target stage


# ------------------------------------------------------------------
# Skill Assessment
# ------------------------------------------------------------------


class SkillItem(BaseModel):
    """A single skill with self-assessed and required levels."""
    name: str
    level: int = 0        # 0-100 self-assessment
    required: int = 0     # 0-100 job requirement
    category: str = ""


class GapItem(BaseModel):
    """A skill gap with improvement suggestion."""
    skill: str
    gap: int              # required - level
    suggestion: str = ""


class SkillAssessmentResponse(BaseModel):
    """Full skill assessment data for a session."""
    session_id: str
    target_role: str = ""
    match_rate: float = 0.0
    skills: List[SkillItem] = []
    gaps: List[GapItem] = []


class SkillAssessmentUpdateRequest(BaseModel):
    """Payload for updating skill assessment data."""
    target_role: Optional[str] = None
    skills: Optional[List[SkillItem]] = None


# ------------------------------------------------------------------
# WebSocket message envelope
# ------------------------------------------------------------------


class WsMessage(BaseModel):
    """Generic WebSocket message envelope.

    Server → Client:
      {"type": "dialogue", "content": "...", "agent": "brain"}
      {"type": "token", "content": "你"}
      {"type": "fragment_break"}
      {"type": "done"}
      {"type": "error", "message": "..."}
      {"type": "status", "phase": "analyzing|responding", "agent": "work_agent"}
      {"type": "demand_analysis", "data": {...}}
      {"type": "pong"}

    Client → Server:
      {"type": "chat", "content": "你好"}
      {"type": "ping"}
    """
    type: str
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    phase: Optional[str] = None
    agent: Optional[str] = None
