"""
StreamingHandler — agent-free streaming chat handler.

Replaces AgentPipeline (BrainAgent → WorkAgent/EmotionAgent routing)
with direct LLMClient.stream_chat() calls through FragmentStreamer.

Maintains the same WebSocket message contract for frontend compatibility:
  dialogue, demand_analysis, status, token, fragment_break, done, error

Emotion detection uses simple keyword matching (no LLM analysis needed).
Session data (turns, mood, profile) is recorded on SessionData directly.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from opencareer.agents.llm_client import LLMClient
from web_api.fragment_streamer import FragmentStreamer
from web_api.session_store import SessionData

logger = logging.getLogger("web_api.streaming_handler")

# ------------------------------------------------------------------
# Keyword lists for lightweight demand/emotion detection
# ------------------------------------------------------------------

_EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "anxiety": ["紧张", "焦虑", "担心", "害怕", "不安", "慌", "nervous", "anxious", "worry", "anxiety"],
    "stress": ["压力", "累", "崩溃", "喘不过气", "stress", "overwhelmed", "exhausted"],
    "frustration": ["沮丧", "失落", "灰心", "失望", "frustrat", "disappoint", "discouraged"],
    "self_doubt": ["不行", "做不到", "不适合", "能力不够", "怀疑", "自卑", "doubt", "not good enough"],
    "sadness": ["难过", "伤心", "哭", "悲伤", "sad", "depress", "unhappy"],
    "loneliness": ["孤独", "一个人", "没人理解", "孤单", "lonely", "alone"],
    "general_encouragement": [],  # fallback
}

_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "resume": ["简历", "CV", "resume", "履历"],
    "interview": ["面试", "interview", "面经"],
    "career_plan": ["职业规划", "方向", "发展", "career plan", "career path"],
    "job_search": ["求职", "找工作", "投递", "job search", "校招", "社招", "实习"],
    "skill_learning": ["学习", "技能", "编程", "learn", "skill", "Python", "Java"],
    "salary": ["薪资", "工资", "薪水", "salary", "compensation", "negotiation"],
    "offer": ["offer", "录用", "offer选择", "对比"],
}

_EMOTION_CATEGORY_TO_MOOD: Dict[str, str] = {
    "anxiety": "anxious",
    "stress": "stressed",
    "frustration": "frustrated",
    "self_doubt": "discouraged",
    "sadness": "discouraged",
    "loneliness": "discouraged",
}

_POSITIVE_MOODS = {"encouraged", "excited", "confident", "hopeful", "motivated", "optimistic", "positive", "calm", "neutral"}


# ------------------------------------------------------------------
# StreamingHandler
# ------------------------------------------------------------------

class StreamingHandler:
    """Handles a single WebSocket session's chat streaming.

    Created once per WebSocket connection. Uses LLMClient directly
    (no BrainAgent/WorkAgent/EmotionAgent) and records all state
    into SessionData.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        session: SessionData,
        system_prompt: str,
    ):
        self.llm = llm_client
        self.session = session
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process_message(
        self,
        user_input: str,
        send_json: Callable[[dict], Any],
    ) -> None:
        """Process a user message through direct LLM streaming.

        WebSocket message sequence (same contract as AgentPipeline):
        1. dialogue — empty placeholder (no BrainAgent in this mode)
        2. demand_analysis — keyword-based emotion/topic detection
        3. status: "responding" — signals streaming start
        4. token / fragment_break — streamed response fragments
        5. done — signals completion
        """
        # Record user turn
        self.session.add_turn("user", user_input)

        # Lightweight keyword-based analysis (no LLM call needed)
        demand_analysis = self._analyze_input(user_input)

        # Record emotion if detected
        if demand_analysis["emotions"]:
            self.session.record_emotion(demand_analysis)

        # Send empty dialogue placeholder (frontend expects this type)
        await send_json({
            "type": "dialogue",
            "content": "",
            "agent": "system",
        })

        # Send demand analysis for sidebar
        await send_json({
            "type": "demand_analysis",
            "data": {
                "demand_type": demand_analysis.get("demand_type", "general"),
                "confidence": demand_analysis.get("confidence", 0.5),
                "target_agent": "streaming_handler",  # no agent routing
                "emotions": [
                    {"label": e} for e in demand_analysis.get("emotions", [])
                ],
                "overall_state": demand_analysis.get("overall_state", "neutral"),
                "support_intensity": demand_analysis.get("support_intensity", "none"),
            },
        })

        # Build context for the LLM
        context = self._build_context(user_input, demand_analysis)

        # Send status
        await send_json({
            "type": "status",
            "phase": "responding",
            "agent": "streaming_handler",
        })

        try:
            full_response = ""
            streamer = FragmentStreamer(send_json)
            async for token in self.llm.stream_chat(
                system_prompt=self.system_prompt,
                user_input=user_input,
                context=context,
            ):
                full_response += token
                await streamer.feed(token)
            await streamer.flush()

            # Record assistant turn
            if full_response:
                self.session.add_turn("assistant", full_response)

            await send_json({"type": "done"})

        except Exception as exc:
            logger.error(f"Streaming error: {exc}")
            await send_json({"type": "error", "message": str(exc)})

    # ------------------------------------------------------------------
    # Lightweight analysis (keyword-based, no LLM)
    # ------------------------------------------------------------------

    def _analyze_input(self, user_input: str) -> Dict[str, Any]:
        """Detect emotions and topics via keyword matching.

        Returns a dict with the same shape as BrainAgent's demand_analysis
        output, so the frontend demand_analysis handler works unchanged.
        """
        input_lower = user_input.lower()

        # Emotion detection
        detected_emotions: List[str] = []
        detected_category: Optional[str] = None
        for category, keywords in _EMOTION_KEYWORDS.items():
            if category == "general_encouragement":
                continue
            for kw in keywords:
                if kw.lower() in input_lower:
                    detected_emotions.append(category)
                    if detected_category is None:
                        detected_category = category
                    break

        # Topic detection
        detected_topics: List[str] = []
        for topic, keywords in _TOPIC_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in input_lower:
                    detected_topics.append(topic)
                    break

        # Determine demand type
        if detected_emotions and detected_topics:
            demand_type = "mixed"
        elif detected_emotions:
            demand_type = "emotional"
        elif detected_topics:
            demand_type = "job_related"
        else:
            demand_type = "general"

        # Determine support intensity
        if detected_category:
            support_intensity = {
                "anxiety": "medium",
                "stress": "high",
                "frustration": "medium",
                "self_doubt": "high",
                "sadness": "high",
                "loneliness": "medium",
            }.get(detected_category, "low")
        else:
            support_intensity = "none"

        # Determine overall state
        if detected_category:
            overall_state = _EMOTION_CATEGORY_TO_MOOD.get(detected_category, "neutral")
        else:
            overall_state = "neutral"

        # Record mood
        if detected_category:
            self.session.record_mood(detected_category, trigger=user_input[:100])

        # Update profile keywords from input
        if detected_topics:
            self.session.add_concern(", ".join(detected_topics))

        return {
            "emotions": [{"label": e} for e in detected_emotions],
            "overall_state": overall_state,
            "support_intensity": support_intensity,
            "demand_type": demand_type,
            "confidence": 0.6 if (detected_emotions or detected_topics) else 0.3,
            "primary_emotion_category": detected_category,
            "topics": detected_topics,
        }

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(self, user_input: str, demand_analysis: Dict[str, Any]) -> str:
        """Build a context string for the LLM stream_chat call.

        Includes: emotion calibration, conversation history, user profile,
        and mood trend data — similar to what agent_pipeline.py built for
        each downstream agent.
        """
        parts: List[str] = []

        # Emotion calibration
        emotions = demand_analysis.get("emotions", [])
        overall_state = demand_analysis.get("overall_state", "")
        support_intensity = demand_analysis.get("support_intensity", "")
        topics = demand_analysis.get("topics", [])

        if emotions:
            emotion_labels = [e["label"] for e in emotions]
            parts.append(f"检测到用户情绪：{', '.join(emotion_labels)}")

        if topics:
            parts.append(f"检测到用户关心的求职话题：{', '.join(topics)}")
        else:
            parts.append("未检测到特定话题，请提供通用的职业发展引导")

        if support_intensity and support_intensity != "none":
            intensity_guide = {
                "low": "用户情绪略有低落，在提供建议时适当加入鼓励语气，但保持工作主线。",
                "medium": "用户有明显情绪波动，在回应前先认可其感受，再温和地引导到求职话题。",
                "high": "用户情绪波动较大，优先共情，降低工作建议的强度，主动询问是否需要情感支持。",
            }.get(support_intensity, "")
            if intensity_guide:
                parts.append(f"情绪校准：{intensity_guide}")

        # Conversation history
        history = self.session.get_formatted_history(max_turns=5)
        if history:
            parts.append(f"近期对话历史：\n{history}")

        # User profile
        profile = self.session.user_profile
        profile_parts = []
        if profile.get("grade_level"):
            profile_parts.append(f"年级：{profile['grade_level']}")
        if profile.get("major"):
            profile_parts.append(f"专业：{profile['major']}")
        if profile.get("school"):
            profile_parts.append(f"学校：{profile['school']}")
        if profile.get("target_role"):
            profile_parts.append(f"目标岗位：{profile['target_role']}")
        if profile.get("background_summary"):
            profile_parts.append(f"背景概述：{profile['background_summary']}")
        if profile_parts:
            parts.append(f"=== 已知用户信息 ===\n" + "\n".join(profile_parts))

        # Mood trends (for emotional context)
        mood_trends = self.session.detect_mood_trends()
        if mood_trends and mood_trends.get("trend") != "insufficient_data":
            parts.append(f"情绪趋势数据：{json.dumps(mood_trends, ensure_ascii=False)}")

        if mood_trends.get("needs_intervention"):
            parts.append("注意：用户近期情绪趋势需要更多关注和关怀")

        return "\n\n".join(parts)
