"""
In-memory session store mapping session_id → SessionData.

SessionData is a lightweight dict-based replacement for ConversationContext,
removing the dependency on the agent architecture while maintaining the same
data-access interface for REST endpoints.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("web_api.session_store")

# ------------------------------------------------------------------
# Emotion keyword mapping (formerly in ConversationContext + agents)
# ------------------------------------------------------------------

_EMOTION_MOOD_MAP: Dict[str, str] = {
    "anxiety": "anxious",
    "anxious": "anxious",
    "stress": "stressed",
    "stressed": "stressed",
    "frustration": "frustrated",
    "frustrated": "frustrated",
    "discouragement": "discouraged",
    "depressed": "discouraged",
    "sadness": "discouraged",
    "sad": "discouraged",
    "fear": "anxious",
    "worry": "anxious",
    "self_doubt": "discouraged",
    "uncertainty": "anxious",
    "confusion": "anxious",
    "nervous": "anxious",
    "disappointment": "discouraged",
    "helplessness": "discouraged",
    "overwhelmed": "stressed",
    "loneliness": "discouraged",
}

_POSITIVE_STATES = {"encouraged", "excited", "confident", "hopeful", "motivated", "optimistic", "positive", "calm", "neutral"}


# ------------------------------------------------------------------
# SessionData
# ------------------------------------------------------------------

class SessionData:
    """Lightweight session-scoped data container.

    Replaces ConversationContext from the agent architecture with
    plain dicts and lists. Maintains the same access patterns so
    REST endpoints work without modification.
    """

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.current_agent: Optional[str] = None
        self.title: str = "新会话"
        self.preview: str = ""

        # Conversation turns: list of {"agent": str, "content": str, "timestamp": str}
        self.history: List[Dict[str, str]] = []

        # Emotion analysis per turn: list of dicts with keys:
        #   emotions, overall_state, support_intensity, demand_type, confidence, timestamp
        self.emotion_history: List[Dict[str, Any]] = []

        # Mood tracking: list of {"category": str, "trigger": str, "timestamp": str}
        self.mood_history: List[Dict[str, str]] = []

        # Job progress: stage -> list of job cards
        # Stages: applied, interviewing, offered, rejected
        # Each card: {"id": str, "company": str, "role": str, "date": str, "note": Optional[str]}
        self.job_progress: Dict[str, List[Dict[str, Any]]] = {
            "applied": [],
            "interviewing": [],
            "offered": [],
            "rejected": [],
        }

        # User profile extracted from conversation
        self.user_profile: Dict[str, Any] = {
            "grade_level": None,
            "major": None,
            "school": None,
            "target_role": None,
            "job_search_stage": None,
            "skill_focus": [],
            "common_concerns": [],
            "background_summary": None,
        }

        # Skill assessment data
        self.skill_assessment: Dict[str, Any] = {
            "target_role": "",
            "skills": [],  # list of {name, level, required, category}
        }

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def add_turn(self, agent: str, content: str) -> None:
        self.current_agent = agent
        self.history.append({
            "agent": agent,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Auto-set title from first user message
        if agent == "user" and self.title == "新会话":
            self.title = content[:30] + ("..." if len(content) > 30 else "")
        # Update preview from latest user message
        if agent == "user":
            self.preview = content[:50] + ("..." if len(content) > 50 else "")

    def get_formatted_history(self, max_turns: int = 5) -> str:
        if not self.history:
            return ""
        recent = self.history[-max_turns:]
        lines = []
        for t in recent:
            role = "用户" if t["agent"] == "user" else f"AI({t['agent']})"
            lines.append(f"{role}: {t['content'][:200]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def update_profile(self, key: str, value: Any) -> None:
        if key in self.user_profile:
            self.user_profile[key] = value

    def add_concern(self, concern: str) -> None:
        concerns = self.user_profile.setdefault("common_concerns", [])
        if concern not in concerns:
            concerns.append(concern)

    # ------------------------------------------------------------------
    # Emotion / Mood
    # ------------------------------------------------------------------

    def record_emotion(self, analysis: Dict[str, Any]) -> None:
        analysis.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self.emotion_history.append(analysis)

    def record_mood(self, category: str, trigger: str = "") -> None:
        self.mood_history.append({
            "category": category,
            "trigger": trigger,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def detect_mood_trends(self) -> Dict[str, Any]:
        """Simplified mood trend detection based on mood_history.

        Returns a dict compatible with what server.py expects.
        """
        if len(self.mood_history) < 2:
            return {
                "trend": "insufficient_data",
                "current_mood": self.mood_history[-1]["category"] if self.mood_history else None,
                "needs_intervention": False,
                "consecutive_low": 0,
            }

        moods = [m["category"] for m in self.mood_history[-10:]]
        english_moods = [_EMOTION_MOOD_MAP.get(m, m) for m in moods]

        # Count negative moods (not in positive set)
        negative_count = sum(1 for m in english_moods if m not in _POSITIVE_STATES)
        current_mood = english_moods[-1]

        # Check consecutive negative
        consecutive = 0
        for m in reversed(english_moods):
            if m not in _POSITIVE_STATES:
                consecutive += 1
            else:
                break

        # Determine trend
        first_half_neg = sum(1 for m in english_moods[:len(english_moods)//2] if m not in _POSITIVE_STATES)
        second_half_neg = sum(1 for m in english_moods[len(english_moods)//2:] if m not in _POSITIVE_STATES)

        if second_half_neg > first_half_neg:
            trend = "declining"
        elif second_half_neg < first_half_neg:
            trend = "improving"
        else:
            trend = "stable"

        needs_intervention = consecutive >= 3

        return {
            "trend": trend,
            "current_mood": current_mood,
            "needs_intervention": needs_intervention,
            "consecutive_low": consecutive,
        }

    def get_accumulated_emotion_assessment(self) -> Dict[str, Any]:
        """Simplified accumulated emotion assessment.

        Returns a dict compatible with what server.py expects.
        """
        if not self.emotion_history:
            return {
                "should_defer_work": False,
                "negative_ratio": 0.0,
                "reason": "",
                "consecutive_negative": 0,
            }

        recent = self.emotion_history[-10:]
        negative_count = 0
        consecutive_negative = 0
        for entry in recent:
            state = entry.get("overall_state", "neutral")
            if state not in _POSITIVE_STATES and state != "neutral":
                negative_count += 1

        # Calculate consecutive from the end
        for entry in reversed(recent):
            state = entry.get("overall_state", "neutral")
            if state not in _POSITIVE_STATES and state != "neutral":
                consecutive_negative += 1
            else:
                break

        negative_ratio = negative_count / len(recent) if recent else 0.0
        should_defer_work = (
            consecutive_negative >= 2
            or negative_ratio > 0.6
            or self.detect_mood_trends().get("needs_intervention", False)
        )

        reason = ""
        if should_defer_work:
            reasons = []
            if consecutive_negative >= 2:
                reasons.append(f"连续{consecutive_negative}轮负面情绪")
            if negative_ratio > 0.6:
                reasons.append(f"负面占比{negative_ratio:.0%}")
            if self.detect_mood_trends().get("needs_intervention"):
                reasons.append("情绪趋势报警")
            reason = "；".join(reasons)

        return {
            "should_defer_work": should_defer_work,
            "negative_ratio": negative_ratio,
            "reason": reason,
            "consecutive_negative": consecutive_negative,
        }


# ------------------------------------------------------------------
# SessionStore
# ------------------------------------------------------------------

class SessionStore:
    """Thread-safe in-memory store for conversation sessions."""

    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}

    def create(self, user_id: str = "default_user") -> dict:
        session_id = uuid.uuid4().hex[:12]
        session = SessionData(user_id=user_id)
        self._sessions[session_id] = session
        logger.info(f"Session created: {session_id} for user={user_id}")
        return {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": session.created_at,
        }

    def get(self, session_id: str) -> Optional[SessionData]:
        return self._sessions.get(session_id)

    def get_info(self, session_id: str) -> Optional[dict]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "title": session.title,
            "preview": session.preview,
            "turn_count": len(session.history),
            "current_agent": session.current_agent,
            "created_at": session.created_at,
        }

    def delete(self, session_id: str) -> bool:
        existed = session_id in self._sessions
        self._sessions.pop(session_id, None)
        if existed:
            logger.info(f"Session deleted: {session_id}")
        return existed

    @property
    def active_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def list_sessions(self) -> list[dict]:
        return [self.get_info(sid) for sid in self._sessions if self.get_info(sid)]


# Singleton
session_store = SessionStore()
