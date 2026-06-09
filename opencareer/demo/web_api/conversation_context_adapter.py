"""
ConversationContextAdapter — wraps SessionData to expose the ConversationContext
interface expected by AgentPipeline / BrainAgent / WorkAgent / EmotionAgent.

Handles 9 key differences between the two data models:
  1. History format:  dict{"agent"} → ConversationTurn.role
  2. record_mood param:  category → mood;  storage: top-level → user_profile
  3. record_emotion → record_emotion_analysis (method rename)
  4. switch_agent (missing → local handoff_stack)
  5. handoff_stack (missing → local list)
  6. resume_data in user_profile (missing → initialised in __init__)
  7. detect_mood_trends(window) — ConversationContext logic on user_profile data
  8. get_accumulated_emotion_assessment(window) — ConversationContext logic
  9. History access via .role attribute instead of ["agent"] key
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from opencareer.agents.conversation_context import ConversationTurn
from opencareer.memory.resume_schema import new_empty_resume_data
from web_api.session_store import SessionData


class ConversationContextAdapter:
    """Wraps a SessionData instance and exposes the full ConversationContext
    interface that AgentPipeline and its agents expect.

    SessionData is the "source of truth" for persistent data (history,
    user_profile, emotion_history).  The adapter adds in-memory-only
    behaviour (handoff_stack, switch_agent) that the agent architecture
    needs but the lightweight session layer doesn't model.
    """

    def __init__(self, session: SessionData):
        self._session = session
        self.logger = logging.getLogger("web_api.conversation_context_adapter")

        # In-memory-only structures (not persisted by SessionData)
        self.handoff_stack: List[Dict[str, Any]] = []

        # Ensure user_profile has the sub-keys that ConversationContext expects.
        # SessionData.user_profile is a plain dict without mood_history or resume_data.
        up = session.user_profile
        if "mood_history" not in up:
            up["mood_history"] = []
        if "resume_data" not in up:
            up["resume_data"] = new_empty_resume_data()

    # ------------------------------------------------------------------
    # Properties (read-through to SessionData)
    # ------------------------------------------------------------------

    @property
    def user_id(self) -> str:
        return self._session.user_id

    @property
    def current_agent(self) -> Optional[str]:
        return self._session.current_agent

    @current_agent.setter
    def current_agent(self, value: Optional[str]) -> None:
        self._session.current_agent = value

    @property
    def emotion_history(self) -> List[Dict[str, Any]]:
        return self._session.emotion_history

    @emotion_history.setter
    def emotion_history(self, value: List[Dict[str, Any]]) -> None:
        self._session.emotion_history = value

    @property
    def user_profile(self) -> Dict[str, Any]:
        """Return SessionData.user_profile (already enhanced with mood_history
        and resume_data in __init__)."""
        return self._session.user_profile

    @user_profile.setter
    def user_profile(self, value: Dict[str, Any]) -> None:
        self._session.user_profile = value

    # ------------------------------------------------------------------
    # History — dict ↔ ConversationTurn adapter
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[ConversationTurn]:
        """Convert SessionData history dicts to ConversationTurn objects.

        SessionData stores  {"agent": str, "content": str, "timestamp": str}
        ConversationTurn has  .role    .content       .timestamp (datetime)
        """
        turns: List[ConversationTurn] = []
        for d in self._session.history:
            ts = d.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    ts = datetime.now()
            elif ts is None:
                ts = datetime.now()
            turns.append(ConversationTurn(
                role=d["agent"],
                content=d["content"],
                timestamp=ts,
            ))
        return turns

    def add_turn(self, role: str, content: str) -> ConversationTurn:
        """Record a conversation turn.  Delegates to SessionData.add_turn
        (which stores dicts) and returns a ConversationTurn for callers
        that expect the ConversationContext interface."""
        # SessionData uses "agent" key; ConversationContext uses "role"
        self._session.add_turn(role, content)  # agent param = role
        ts = datetime.now()
        return ConversationTurn(role=role, content=content, timestamp=ts)

    def get_history_for_context(self, max_turns: int = 10) -> List[ConversationTurn]:
        """Return the most-recent max_turns as ConversationTurn objects."""
        history = self.history  # full list via property
        if len(history) <= max_turns:
            return history
        return history[-max_turns:]

    def get_formatted_history(self, max_turns: int = 10) -> str:
        """Format recent history for LLM prompts (English labels, full content).

        Mirrors ConversationContext.get_formatted_history exactly:
          - "User" for user turns
          - "Agent (role)" for agent turns
          - Full content, no truncation
        """
        turns = self.get_history_for_context(max_turns)
        lines: List[str] = []
        for t in turns:
            label = "User" if t.role == "user" else f"Agent ({t.role})"
            lines.append(f"{label}: {t.content}")
        return "\n".join(lines)

    def clear_history(self) -> None:
        """Clear all conversation history (profile data is preserved)."""
        self._session.history.clear()
        self.logger.info("Conversation history cleared")

    # ------------------------------------------------------------------
    # Agent handoff (in-memory only — SessionData has no handoff model)
    # ------------------------------------------------------------------

    def switch_agent(self, agent_name: str, reason: str = "") -> None:
        """Switch the active agent, pushing the previous onto handoff_stack."""
        previous = self._session.current_agent
        self._session.current_agent = agent_name

        self.handoff_stack.append({
            "from": previous,
            "to": agent_name,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })
        self.logger.info(
            f"Agent switch: {previous} -> {agent_name} | reason={reason}"
        )

    def get_current_agent(self) -> Optional[str]:
        """Return the currently-active agent name."""
        return self._session.current_agent

    def get_handoff_chain(self) -> List[Dict[str, Any]]:
        """Return the full handoff history."""
        return list(self.handoff_stack)

    # ------------------------------------------------------------------
    # Mood tracking — ConversationContext logic on user_profile data
    # ------------------------------------------------------------------

    def record_mood(self, mood: str, trigger: str = "") -> None:
        """Record a mood entry.

        Dual-writes so both the ConversationContext consumers (which read
        from user_profile["mood_history"]) and the REST endpoints (which
        read from session.mood_history) see the data.

        The SessionData version uses {"category": …} keys; the
        ConversationContext version uses {"mood": …}.  We store both
        formats in the respective locations.
        """
        now_iso = datetime.now().isoformat()

        # Write to user_profile (ConversationContext format)
        self._session.user_profile["mood_history"].append({
            "mood": mood,
            "timestamp": now_iso,
            "trigger": trigger,
        })

        # Write to session-level mood_history (SessionData / REST format)
        self._session.mood_history.append({
            "category": mood,
            "trigger": trigger,
            "timestamp": now_iso,
        })

        self.logger.debug(f"Mood recorded: {mood} | trigger={trigger}")

    def detect_mood_trends(self, window: int = 5) -> Dict[str, Any]:
        """Detect mood trends from user_profile["mood_history"].

        Uses ConversationContext logic (reads "mood" key, low_moods set,
        window parameter) rather than SessionData logic (which reads
        "category" key and has no window param).
        """
        mood_history: List[Dict[str, Any]] = (
            self._session.user_profile.get("mood_history") or []
        )

        if not mood_history:
            return {
                "trend": "insufficient_data",
                "current_mood": None,
                "consecutive_low": 0,
                "needs_intervention": False,
            }

        low_moods = {"stressed", "anxious", "discouraged"}

        recent = (
            mood_history[-window:]
            if len(mood_history) > window
            else mood_history
        )
        current = recent[-1]["mood"]

        # Count consecutive low moods (from most recent backward)
        consecutive_low = 0
        for entry in reversed(mood_history):
            if entry["mood"] in low_moods:
                consecutive_low += 1
            else:
                break

        # Simple trend detection over the window
        if len(recent) >= 3:
            recent_moods = [e["mood"] for e in recent[-3:]]
            lows = sum(1 for m in recent_moods if m in low_moods)

            if lows >= 3:
                trend = "declining"
            elif lows == 0:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "stable"

        needs_intervention = consecutive_low >= 3 or trend == "declining"

        return {
            "trend": trend,
            "current_mood": current,
            "consecutive_low": consecutive_low,
            "needs_intervention": needs_intervention,
        }

    # ------------------------------------------------------------------
    # Emotion analysis
    # ------------------------------------------------------------------

    def record_emotion_analysis(self, analysis: Dict[str, Any]) -> None:
        """Record per-turn emotion analysis (ConversationContext name).

        Delegates to SessionData.record_emotion (the SessionData method
        name), which stores to session.emotion_history.
        """
        self._session.record_emotion({
            "emotions": list(analysis.get("emotions", [])),
            "overall_state": analysis.get("overall_state", "neutral"),
            "support_intensity": analysis.get("support_intensity", "none"),
            "demand_type": analysis.get("demand_type", "unknown"),
            "confidence": analysis.get("confidence", 0.0),
            "timestamp": datetime.now().isoformat(),
        })

    def get_accumulated_emotion_assessment(
        self, window: int = 5
    ) -> Dict[str, Any]:
        """Assess accumulated emotional state across recent turns.

        Uses ConversationContext logic: negative_states = {"negative", "crisis"},
        calls self.detect_mood_trends(window=window), returns trend_result.
        """
        trend_result = self.detect_mood_trends(window=window)

        emotion_history: List[Dict[str, Any]] = self._session.emotion_history
        recent = (
            emotion_history[-window:]
            if len(emotion_history) > window
            else emotion_history
        )

        if not recent:
            return {
                "negative_ratio": 0.0,
                "consecutive_negative": 0,
                "should_defer_work": False,
                "reason": "尚无情感历史记录",
                "trend_result": trend_result,
            }

        negative_states = {"negative", "crisis"}
        consecutive_negative = 0
        negative_count = 0

        for entry in reversed(recent):
            if entry.get("overall_state") in negative_states:
                consecutive_negative += 1
            else:
                break

        for entry in recent:
            if entry.get("overall_state") in negative_states:
                negative_count += 1

        negative_ratio = negative_count / len(recent)

        should_defer = (
            consecutive_negative >= 2
            or negative_ratio > 0.6
            or trend_result.get("needs_intervention", False)
        )

        if should_defer:
            reasons: List[str] = []
            if consecutive_negative >= 2:
                reasons.append(f"连续{consecutive_negative}轮情绪负面")
            if negative_ratio > 0.6:
                reasons.append(f"负面情绪占比{negative_ratio:.0%}")
            if trend_result.get("needs_intervention"):
                reasons.append(trend_result.get("trend", "负面趋势"))
            reason = "；".join(reasons)
        else:
            reason = "情绪状态正常"

        return {
            "negative_ratio": negative_ratio,
            "consecutive_negative": consecutive_negative,
            "should_defer_work": should_defer,
            "reason": reason,
            "trend_result": trend_result,
        }

    # ------------------------------------------------------------------
    # User profile helpers
    # ------------------------------------------------------------------

    def update_profile(self, key: str, value: Any) -> None:
        """Update a field in the user profile."""
        if key in self._session.user_profile:
            self._session.user_profile[key] = value
            self.logger.debug(f"Profile updated: {key} = {value}")
        else:
            self.logger.warning(f"Unknown profile field: {key}")

    def add_concern(self, concern: str) -> None:
        """Add a common concern if not already present."""
        concerns = self._session.user_profile.setdefault("common_concerns", [])
        if concern not in concerns:
            concerns.append(concern)
            self.logger.debug(f"Concern added: {concern}")
        # Also delegate to SessionData.add_concern for any extra behaviour
        self._session.add_concern(concern)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full context (ConversationContext format)."""
        return {
            "user_id": self._session.user_id,
            "current_agent": self._session.current_agent,
            "history": [
                {
                    "role": d["agent"],
                    "content": d["content"],
                    "timestamp": d.get("timestamp", ""),
                }
                for d in self._session.history
            ],
            "handoff_stack": list(self.handoff_stack),
            "emotion_history": list(self._session.emotion_history),
            "user_profile": {
                "mood_history": list(
                    self._session.user_profile.get("mood_history", [])
                ),
                "common_concerns": list(
                    self._session.user_profile.get("common_concerns", [])
                ),
                "job_search_stage": self._session.user_profile.get(
                    "job_search_stage"
                ),
                "target_role": self._session.user_profile.get("target_role"),
                "skill_focus": list(
                    self._session.user_profile.get("skill_focus", [])
                ),
                "grade_level": self._session.user_profile.get("grade_level"),
                "major": self._session.user_profile.get("major"),
                "school": self._session.user_profile.get("school"),
                "background_summary": self._session.user_profile.get(
                    "background_summary"
                ),
                "resume_data": dict(
                    self._session.user_profile.get("resume_data", {})
                ),
            },
        }

    def __repr__(self) -> str:
        return (
            f"ConversationContextAdapter(user_id={self._session.user_id}, "
            f"current_agent={self._session.current_agent}, "
            f"turns={len(self._session.history)}, "
            f"handoffs={len(self.handoff_stack)})"
        )
