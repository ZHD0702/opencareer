"""
Agent Pipeline for OpenCareer Web API.

Wires BrainAgent → WorkAgent/EmotionAgent through the WebSocket pipeline
with streaming support. Follows the judge-and-handoff pattern from
quick_dialogue.py: BrainAgent analyzes demand, determines target agent,
then the pipeline streams the downstream agent's response token-by-token.

Key design decisions:
- BrainAgent handles ALL context side effects (user turn, emotion analysis,
  profile extraction, brain turn recording) via process_user_request()
- Pipeline replicates each downstream agent's context-building logic and
  calls llm_client.stream_chat() directly — avoids double-recording that
  would occur if we called the downstream agents' process_user_request()
- After streaming completes, pipeline records the downstream agent's turn
"""

import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, Optional

from opencareer.agents.brain.brain_agent import BrainAgent
from opencareer.agents.work_agent.work_agent import WorkAgent
from opencareer.agents.emotion_agent.emotion_agent import EmotionAgent
from opencareer.agents.llm_client import LLMClient
from opencareer.agents.conversation_context import ConversationContext
from opencareer.agents.persona import get_agent_persona
from opencareer.prompts.registry import get_global_registry
from web_api.fragment_streamer import FragmentStreamer

_registry = get_global_registry()
logger = logging.getLogger("web_api.agent_pipeline")

# Load keyword lists from YAML (shared with agents)
_EMOTION_KEYWORDS: Dict[str, list] = _registry.get("dialogues.support.emotion_keywords", {})


class AgentPipeline:
    """Orchestrates BrainAgent → downstream agent with streaming responses.

    Created once per WebSocket session. Holds references to all three agents
    with a shared ConversationContext, following the quick_dialogue.py pattern.
    """

    def __init__(self, llm_client: LLMClient, ctx: ConversationContext):
        self.llm_client = llm_client
        self.ctx = ctx

        # Create agents with shared context (matching quick_dialogue.py setup)
        self.brain = BrainAgent(
            llm_client=llm_client,
            deepseek_api_key=None,  # LLM client already has the key
            conversation_context=ctx,
        )

        self.work_agent = WorkAgent(
            llm_client=llm_client,
            mcp_server_url="http://localhost:8000",
            conversation_context=ctx,
        )

        self.emotion_agent = EmotionAgent(
            llm_client=llm_client,
            mcp_server_url="http://localhost:8000",
            conversation_context=ctx,
        )

        # Mock MCP skill call (MCP server not connected in web mode)
        self.work_agent._call_skill_via_mcp = self._mock_call_skill_via_mcp
        self.emotion_agent._call_emotion_support_skill = self._mock_call_skill_via_mcp

        # Mock agent registry so BrainAgent can resolve target agent names
        class MockAgentRegistry:
            def get_agent(self, name):
                agents = {
                    "work_agent": self.work_agent,
                    "emotion_agent": self.emotion_agent,
                }
                return agents.get(name)

        self.brain.agent_registry = MockAgentRegistry()

        # Cache persona prompts for streaming
        self._emotion_prompt = _registry.get("system.emotion_agent_prompt.system_prompt", "")
        if not self._emotion_prompt:
            self._emotion_prompt = self.emotion_agent.system_prompt

        logger.info("AgentPipeline initialized with Brain + Work + Emotion agents")

    # ------------------------------------------------------------------
    # MCP mock
    # ------------------------------------------------------------------

    @staticmethod
    async def _mock_call_skill_via_mcp(skill_name: str, user_input: str, context=None) -> dict:
        return {
            "skill_used": skill_name,
            "result": f"[MCP服务未连接] {skill_name} 暂不可用，请连接到MCP服务器后再试",
            "error": "MCP server not connected",
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process_message(
        self,
        user_input: str,
        send_json: Callable[[dict], Any],
    ) -> None:
        """Process a user message through the agent pipeline with streaming.

        1. BrainAgent analyzes demand → records context → determines target
        2. Sends brain's dialogue response + demand_analysis to client
        3. Streams downstream agent response token-by-token
        4. Records downstream agent turn in context

        Args:
            user_input: User's input text
            send_json: Async callable to send WebSocket JSON messages
        """
        # Step 1: BrainAgent analyzes demand and handles ALL context recording
        brain_result = await self.brain.process_user_request(user_input)

        target_agent = brain_result.get("target_agent")
        demand_analysis = brain_result.get("demand_analysis", {})
        brain_response = brain_result.get("response", "")

        # Step 2: Send brain's dialogue response
        await send_json({
            "type": "dialogue",
            "content": brain_response,
            "agent": "brain",
        })

        # Step 3: Send demand analysis for frontend status display
        await send_json({
            "type": "demand_analysis",
            "data": {
                "demand_type": demand_analysis.get("demand_type", "unknown"),
                "confidence": demand_analysis.get("confidence", 0.0),
                "target_agent": target_agent,
                "emotions": demand_analysis.get("emotions", []),
                "overall_state": demand_analysis.get("overall_state", ""),
                "support_intensity": demand_analysis.get("support_intensity", ""),
            },
        })

        # Step 4: If no target agent (clarification needed), brain handles it
        if target_agent is None:
            await send_json({"type": "done"})
            return

        # Step 5: Stream response from the target downstream agent
        await send_json({
            "type": "status",
            "phase": "responding",
            "agent": target_agent,
        })

        try:
            if target_agent == "work_agent":
                await self._stream_work_agent(user_input, demand_analysis, send_json)
            elif target_agent == "emotion_agent":
                await self._stream_emotion_agent(user_input, demand_analysis, send_json)

            await send_json({"type": "done"})

        except Exception as exc:
            logger.error(f"Agent pipeline streaming error: {exc}")
            await send_json({"type": "error", "message": str(exc)})

    # ------------------------------------------------------------------
    # WorkAgent streaming — replicates _build_conversational_advice()
    # ------------------------------------------------------------------

    async def _stream_work_agent(
        self,
        user_input: str,
        demand_analysis: dict,
        send_json: Callable[[dict], Any],
    ) -> None:
        """Stream WorkAgent response token-by-token.

        Replicates WorkAgent._build_conversational_advice() context-building:
        topic detection, support_intensity calibration, history, profile info.
        """
        input_lower = user_input.lower()

        # Load topic detection keywords
        topic_keywords = _registry.get("dialogues.career_advice.topic_detection_keywords", {})

        topics = []
        for topic, keywords in topic_keywords.items():
            eng_match = any(word in input_lower for word in keywords)
            cn_match = any(word in user_input for word in keywords)
            if eng_match or cn_match:
                topics.append(topic)

        # Update profile from input (mirrors work_agent._build_conversational_advice)
        if topics:
            self.ctx.add_concern(", ".join(topics))
        prefixes = _registry.get("dialogues.career_advice.target_role_prefixes", [])
        for prefix in prefixes:
            if prefix in user_input:
                self.ctx.update_profile("target_role", f"mentioned in: {user_input[-80:]}")

        # Build streaming context
        context_parts = []
        if topics:
            context_parts.append(f"检测到用户关心的求职话题：{', '.join(topics)}")
        else:
            context_parts.append("未检测到特定话题，请提供通用的职业发展引导")

        # Support intensity calibration
        support_intensity = demand_analysis.get("support_intensity", "")
        if support_intensity:
            intensity_guide = {
                "low": "用户情绪略有低落，在提供建议时适当加入鼓励语气，但保持工作主线。",
                "medium": "用户有明显情绪波动，在回应前先认可其感受，再温和地引导到求职话题。",
                "high": "用户情绪波动较大，优先共情，降低工作建议的强度，主动询问是否需要情感支持。",
                "crisis": "用户处于极度困扰中，立即暂停工作建议，主动提出转接情感支持伙伴。",
            }.get(support_intensity, "")
            if intensity_guide:
                context_parts.append(f"情绪校准：{intensity_guide}")

        # Conversation history
        history = self.ctx.get_formatted_history(max_turns=5)
        if history:
            context_parts.append(f"近期对话历史：\n{history}")

        # User profile
        profile = self.ctx.user_profile
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
            context_parts.append(f"=== 已知用户信息 ===\n" + "\n".join(profile_parts))

        context_str = "\n\n".join(context_parts)

        # Stream via LLM
        full_response = ""
        streamer = FragmentStreamer(send_json)
        async for token in self.llm_client.stream_chat(
            system_prompt=self.work_agent.system_prompt,
            user_input=user_input,
            context=context_str,
        ):
            full_response += token
            await streamer.feed(token)
        await streamer.flush()

        # Record the downstream agent's turn
        if full_response:
            self.ctx.add_turn("work_agent", full_response)

    # ------------------------------------------------------------------
    # EmotionAgent streaming — replicates _build_support_response()
    # ------------------------------------------------------------------

    async def _stream_emotion_agent(
        self,
        user_input: str,
        demand_analysis: dict,
        send_json: Callable[[dict], Any],
    ) -> None:
        """Stream EmotionAgent response token-by-token.

        Replicates EmotionAgent._build_support_response() context-building:
        emotion detection, mood recording, support calibration, history, trends.
        """
        # Detect emotion category (mirrors EmotionAgent._detect_emotion)
        emotion_category = self._detect_emotion(user_input, demand_analysis)

        # Record mood in context
        self.ctx.record_mood(emotion_category, trigger=user_input[:100])

        # Check for intervention
        needs_intervention = False
        trends = self.ctx.detect_mood_trends()
        needs_intervention = trends.get("needs_intervention", False)

        # Build streaming context
        context_parts = [f"检测到情绪类别：{emotion_category}"]

        emotions = demand_analysis.get("emotions", [])
        overall_state = demand_analysis.get("overall_state", "")
        support_intensity = demand_analysis.get("support_intensity", "")

        if emotions:
            emotion_labels = [e.get("label", "") for e in emotions if isinstance(e, dict)]
            if emotion_labels:
                context_parts.append(f"情绪分析：{', '.join(emotion_labels)}")
        if overall_state:
            context_parts.append(f"整体状态：{overall_state}")
        if support_intensity:
            context_parts.append(f"支持强度：{support_intensity}")

        if needs_intervention:
            context_parts.append("注意：用户近期情绪趋势需要更多关注和关怀")

        # Conversation history
        history = self.ctx.get_formatted_history(max_turns=5)
        if history:
            context_parts.append(f"近期对话：\n{history}")

        # Mood trend data
        mood_trends = self.ctx.detect_mood_trends()
        if mood_trends:
            context_parts.append(f"情绪趋势数据：{json.dumps(mood_trends, ensure_ascii=False)}")

        context_str = "\n\n".join(context_parts)

        # Use task-specific system prompt if available, fallback to persona
        system_prompt = self._emotion_prompt or self.emotion_agent.system_prompt

        # Stream via LLM
        full_response = ""
        streamer = FragmentStreamer(send_json)
        async for token in self.llm_client.stream_chat(
            system_prompt=system_prompt,
            user_input=user_input,
            context=context_str,
        ):
            full_response += token
            await streamer.feed(token)
        await streamer.flush()

        # Record the downstream agent's turn
        if full_response:
            self.ctx.add_turn("emotion_agent", full_response)

    # ------------------------------------------------------------------
    # Emotion detection (mirrors EmotionAgent._detect_emotion)
    # ------------------------------------------------------------------

    def _detect_emotion(self, user_input: str, demand_analysis: dict) -> str:
        """Detect the primary emotion category from user input.

        Mirrors EmotionAgent._detect_emotion() keyword-matching logic.
        """
        input_lower = user_input.lower()

        for category, triggers in _EMOTION_KEYWORDS.items():
            for trigger in triggers:
                if trigger in input_lower:
                    return category

        # Check demand_analysis for emotional state hints
        emotions = demand_analysis.get("emotions", [])
        if emotions:
            for e in emotions:
                if isinstance(e, dict):
                    label = e.get("label", "").lower()
                    for category in _EMOTION_KEYWORDS.keys():
                        if category in label or category == label:
                            return category

        return "general_encouragement"
