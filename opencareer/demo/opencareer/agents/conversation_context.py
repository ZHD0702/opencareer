"""
Conversation Context for the OpenCareer multi-agent system.

Provides shared conversation state across all agents:
- ConversationTurn: A single turn in a conversation (user or agent message)
- ConversationContext: Full conversation state with history, mood tracking,
  user profile, and handoff tracking.

This is the bridge between the old "structured Dict return" pattern and the
new "dialogue" pattern where agents own conversations and maintain context.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..memory.resume_schema import new_empty_resume_data


@dataclass
class ConversationTurn:
    """A single turn in a conversation.

    Attributes:
        role: "user" or the agent name (e.g. "brain", "work_agent", "emotion_agent")
        content: The message text
        timestamp: When this turn occurred
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = datetime.now()
        elif ts is None:
            ts = datetime.now()
        return cls(role=data["role"], content=data["content"], timestamp=ts)


class ConversationContext:
    """Shared conversation state across all agents.

    Tracks:
    - Conversation history (list of ConversationTurn)
    - Current active agent (which agent is currently handling the conversation)
    - User profile (mood history, common concerns)
    - Handoff state (who handed off to whom, when)
    """

    def __init__(self, user_id: str = "default_user"):
        self.user_id: str = user_id
        self.history: List[ConversationTurn] = []
        self.current_agent: Optional[str] = None
        self.handoff_stack: List[Dict[str, Any]] = []
        self.emotion_history: List[Dict[str, Any]] = []  # per-turn emotion analysis results
        self.logger = logging.getLogger("conversation.context")

        # User profile — built over time as the conversation progresses
        self.user_profile: Dict[str, Any] = {
            "mood_history": [],       # list of {"mood": str, "timestamp": str, "trigger": str}
            "common_concerns": [],    # recurring topics the user brings up
            "job_search_stage": None, # starting / applying / interviewing / waiting / negotiating / accepted
            "target_role": None,      # e.g. "backend engineer"
            "skill_focus": [],        # skills the user wants to develop
            "grade_level": None,      # e.g. "大一", "大二", "大三", "大四", "研一", etc.
            "major": None,            # e.g. "软件工程", "计算机科学"
            "school": None,           # e.g. "北京大学"
            "background_summary": None, # free-text summary of known user info

            # Full resume data structure (Schema + Data 双层结构)
            # Schema: memory.resume_schema defines all ~30 fields across 7 modules
            # Data:  resume_data holds per-user values, extracted passively from
            #        conversation via LLM-based extraction
            "resume_data": new_empty_resume_data(),
        }

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> ConversationTurn:
        """Record one turn in the conversation.

        Args:
            role: "user" or the agent name (e.g. "brain", "work_agent")
            content: The message text

        Returns:
            The created ConversationTurn
        """
        turn = ConversationTurn(role=role, content=content)
        self.history.append(turn)
        self.logger.debug(f"Turn added | role={role} | len(content)={len(content)}")
        return turn

    def get_history_for_context(self, max_turns: int = 10) -> List[ConversationTurn]:
        """Get the most recent turns for LLM context window.

        Args:
            max_turns: Maximum number of recent turns to return

        Returns:
            List of recent ConversationTurn objects
        """
        if len(self.history) <= max_turns:
            return list(self.history)
        return self.history[-max_turns:]

    def get_formatted_history(self, max_turns: int = 10) -> str:
        """Get recent history as a formatted string for inclusion in prompts.

        Args:
            max_turns: Maximum number of recent turns

        Returns:
            Formatted conversation string like "user: ...\nagent: ..."
        """
        turns = self.get_history_for_context(max_turns)
        lines = []
        for t in turns:
            label = "User" if t.role == "user" else f"Agent ({t.role})"
            lines.append(f"{label}: {t.content}")
        return "\n".join(lines)

    def clear_history(self) -> None:
        """Clear all conversation history (profile data is preserved)."""
        self.history.clear()
        self.logger.info("Conversation history cleared")

    # ------------------------------------------------------------------
    # Agent handoff
    # ------------------------------------------------------------------

    def switch_agent(self, agent_name: str, reason: str = "") -> None:
        """Switch the active agent handling the conversation.

        Pushes the previous agent onto the handoff stack so the system
        can trace the full handoff chain.

        Args:
            agent_name: Name of the agent taking over
            reason: Why the handoff is happening
        """
        previous = self.current_agent
        self.current_agent = agent_name

        handoff_record = {
            "from": previous,
            "to": agent_name,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        self.handoff_stack.append(handoff_record)

        self.logger.info(f"Agent switch: {previous} -> {agent_name} | reason={reason}")

    def get_current_agent(self) -> Optional[str]:
        """Get the name of the currently active agent.

        Returns:
            Agent name, or None if no agent is active
        """
        return self.current_agent

    def get_handoff_chain(self) -> List[Dict[str, Any]]:
        """Get the full handoff history for traceability.

        Returns:
            List of handoff records
        """
        return list(self.handoff_stack)

    # ------------------------------------------------------------------
    # Mood tracking
    # ------------------------------------------------------------------

    def record_mood(self, mood: str, trigger: str = "") -> None:
        """Record the user's emotional state at this point in time.

        Args:
            mood: Emotional state label (happy / neutral / stressed / anxious / discouraged / confident)
            trigger: What caused this mood (e.g. "interview rejected", "received offer")
        """
        self.user_profile["mood_history"].append({
            "mood": mood,
            "timestamp": datetime.now().isoformat(),
            "trigger": trigger,
        })
        self.logger.debug(f"Mood recorded: {mood} | trigger={trigger}")

    def detect_mood_trends(self, window: int = 5) -> Dict[str, Any]:
        """Detect mood trends over the recent history.

        Args:
            window: Number of recent mood entries to analyze

        Returns:
            Dict with trend info:
            - trend: "improving" / "declining" / "stable" / "insufficient_data"
            - current_mood: The most recent mood label
            - consecutive_low: Number of consecutive low-mood entries
            - needs_intervention: True if trend suggests stronger support needed
        """
        mood_history = self.user_profile["mood_history"]
        if not mood_history:
            return {
                "trend": "insufficient_data",
                "current_mood": None,
                "consecutive_low": 0,
                "needs_intervention": False,
            }

        # Low-mood categories
        low_moods = {"stressed", "anxious", "discouraged"}

        recent = mood_history[-window:] if len(mood_history) > window else mood_history
        current = recent[-1]["mood"]

        # Count consecutive low moods (from most recent backward)
        consecutive_low = 0
        for entry in reversed(mood_history):
            if entry["mood"] in low_moods:
                consecutive_low += 1
            else:
                break

        # Simple trend detection
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
    # Emotion history tracking (per-turn from BrainAgent analysis)
    # ------------------------------------------------------------------

    def record_emotion_analysis(self, analysis: Dict[str, Any]) -> None:
        """Record per-turn emotion analysis results for accumulated state tracking.

        Stores the full emotion analysis output from BrainAgent so that
        routing decisions can consider emotional trends across turns.

        Args:
            analysis: Normalized demand_analysis dict with emotion fields
                      (emotions, overall_state, support_intensity, demand_type, confidence)
        """
        self.emotion_history.append({
            "emotions": list(analysis.get("emotions", [])),
            "overall_state": analysis.get("overall_state", "neutral"),
            "support_intensity": analysis.get("support_intensity", "none"),
            "demand_type": analysis.get("demand_type", "unknown"),
            "confidence": analysis.get("confidence", 0.0),
            "timestamp": datetime.now().isoformat(),
        })
        self.logger.debug(
            f"Emotion analysis recorded: overall_state={analysis.get('overall_state')}, "
            f"support_intensity={analysis.get('support_intensity')}"
        )

    def get_accumulated_emotion_assessment(self, window: int = 5) -> Dict[str, Any]:
        """Assess accumulated emotional state across recent turns.

        Used by BrainAgent to determine if a user who is asking for work
        tasks is actually in a suitable emotional state. Considers both
        the structured emotion_history and the mood_history trends.

        Args:
            window: Number of recent emotion entries to consider

        Returns:
            Dict with:
            - negative_ratio: proportion of negative/crisis states in window
            - consecutive_negative: count of consecutive negative states
            - should_defer_work: True if accumulated emotion suggests
              routing to emotion_agent instead of work_agent
            - reason: human-readable explanation in Chinese
            - trend_result: result from detect_mood_trends()
        """
        # Check mood_history trends first (from record_mood calls)
        trend_result = self.detect_mood_trends(window=window)

        # Check emotion_history for negative patterns
        recent = (
            self.emotion_history[-window:]
            if len(self.emotion_history) > window
            else self.emotion_history
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

        # Defer work if:
        # 1. consecutive_negative >= 2 (persistent negative across turns)
        # 2. negative_ratio > 0.6 in the window
        # 3. detect_mood_trends says needs_intervention
        should_defer = (
            consecutive_negative >= 2
            or negative_ratio > 0.6
            or trend_result.get("needs_intervention", False)
        )

        if should_defer:
            reasons = []
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
    # User profile
    # ------------------------------------------------------------------

    def update_profile(self, key: str, value: Any) -> None:
        """Update a field in the user profile.

        Args:
            key: Profile field name
            value: New value
        """
        if key in self.user_profile:
            self.user_profile[key] = value
            self.logger.debug(f"Profile updated: {key} = {value}")
        else:
            self.logger.warning(f"Unknown profile field: {key}")

    def add_concern(self, concern: str) -> None:
        """Add a common concern if it's not already in the list.

        Args:
            concern: The concern topic
        """
        if concern not in self.user_profile["common_concerns"]:
            self.user_profile["common_concerns"].append(concern)
            self.logger.debug(f"Concern added: {concern}")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full context to a dictionary.

        This is the bridge to the existing ``context: Dict[str, Any]``
        parameter in ``BaseAgent.process_user_request(user_input, context)``.
        """
        return {
            "user_id": self.user_id,
            "current_agent": self.current_agent,
            "history": [t.to_dict() for t in self.history],
            "handoff_stack": list(self.handoff_stack),
            "emotion_history": list(self.emotion_history),
            "user_profile": {
                "mood_history": list(self.user_profile["mood_history"]),
                "common_concerns": list(self.user_profile["common_concerns"]),
                "job_search_stage": self.user_profile["job_search_stage"],
                "target_role": self.user_profile["target_role"],
                "skill_focus": list(self.user_profile["skill_focus"]),
                "grade_level": self.user_profile["grade_level"],
                "major": self.user_profile["major"],
                "school": self.user_profile["school"],
                "background_summary": self.user_profile["background_summary"],
                "resume_data": dict(self.user_profile.get("resume_data", {})),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Deserialize from a dictionary.

        Args:
            data: Dictionary representation of the context

        Returns:
            A new ConversationContext instance
        """
        ctx = cls(user_id=data.get("user_id", "default_user"))
        ctx.current_agent = data.get("current_agent")
        ctx.history = [ConversationTurn.from_dict(t) for t in data.get("history", [])]
        ctx.handoff_stack = list(data.get("handoff_stack", []))
        ctx.emotion_history = list(data.get("emotion_history", []))

        profile = data.get("user_profile", {})
        ctx.user_profile["mood_history"] = list(profile.get("mood_history", []))
        ctx.user_profile["common_concerns"] = list(profile.get("common_concerns", []))
        ctx.user_profile["job_search_stage"] = profile.get("job_search_stage")
        ctx.user_profile["target_role"] = profile.get("target_role")
        ctx.user_profile["skill_focus"] = list(profile.get("skill_focus", []))
        ctx.user_profile["grade_level"] = profile.get("grade_level")
        ctx.user_profile["major"] = profile.get("major")
        ctx.user_profile["school"] = profile.get("school")
        ctx.user_profile["background_summary"] = profile.get("background_summary")

        # Restore resume_data — merge with empty template to handle schema evolution
        saved_resume = profile.get("resume_data", {})
        if isinstance(saved_resume, dict) and saved_resume:
            template = new_empty_resume_data()
            template.update(saved_resume)
            ctx.user_profile["resume_data"] = template

        return ctx

    def __repr__(self) -> str:
        return (
            f"ConversationContext(user_id={self.user_id}, "
            f"current_agent={self.current_agent}, "
            f"turns={len(self.history)}, "
            f"handoffs={len(self.handoff_stack)})"
        )
