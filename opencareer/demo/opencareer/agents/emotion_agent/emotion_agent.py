"""
Emotion Agent for OpenCareer system.

This agent provides emotional support, encouragement, and stress management
for job seekers during their career development journey. Uses the shared
persona system for consistent dialogue style and ConversationContext for
mood tracking and multi-turn conversation coherence.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from ..base_agent import BaseAgent, AgentMessage
from ..conversation_context import ConversationContext
from ..llm_client import LLMClient
from ..persona import get_agent_persona
from opencareer.prompts.registry import get_global_registry

_registry = get_global_registry()

# Keyword lists loaded from YAML (dialogues/support.yaml)
_EMOTION_KEYWORDS: Dict[str, List[str]] = _registry.get("dialogues.support.emotion_keywords", {})
_JOB_KEYWORDS: List[str] = _registry.get("dialogues.support.job_keywords", [])


class EmotionAgent(BaseAgent):
    """Emotion Agent responsible for emotional support and encouragement.

    Conversational agent that:
    - Engages in persona-driven Chinese dialogue for emotional support
    - Detects and tracks user emotions via ConversationContext
    - Offers handoff to WorkAgent if job-related topics are mentioned
    - Uses MCP emotion_support skill when available, falls back to LLM
    """

    def __init__(self, llm_client: Optional[LLMClient] = None,
                 mcp_server_url: str = "http://localhost:8000",
                 conversation_context: ConversationContext = None):
        """Initialize Emotion Agent.

        Args:
            llm_client: Shared DeepSeek API client for LLM generation
            mcp_server_url: URL of the MCP Server (for emotion_support SKILL)
            conversation_context: Shared conversation state across agents
        """
        super().__init__(
            name="emotion_agent",
            description="Empathetic companion providing emotional support, stress management, and encouragement for job seekers"
        )

        self.llm_client = llm_client
        self.mcp_server_url = mcp_server_url
        self.conversation_context = conversation_context

        # Emotion detection triggers (Chinese + English)
        self.emotion_triggers = _EMOTION_KEYWORDS

        # Register capabilities
        self.capabilities = [
            "emotional_support",
            "stress_management",
            "motivation_boost",
            "confidence_building",
            "work_life_balance_advice"
        ]

        # Cache persona system prompt for consistent dialogue
        self.system_prompt = get_agent_persona("emotion_agent")

        self.logger = logging.getLogger("agent.emotion_agent")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def _initialize(self) -> None:
        """Initialize connection to MCP Server for emotion_support SKILL."""
        self.logger.info(f"Initializing Emotion Agent with MCP Server: {self.mcp_server_url}")

        # Check MCP Server availability (optional for emotion agent)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.mcp_server_url}/health", timeout=5) as response:
                    if response.status == 200:
                        self.logger.info("MCP Server is available for emotion_support SKILL")
                        self.capabilities.append("advanced_emotion_support")
                    else:
                        self.logger.info("MCP Server not available, using basic emotion support")
        except Exception:
            self.logger.info("MCP Server not available, using basic emotion support")

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    def _get_context(self, context: Dict[str, Any] = None) -> Optional[ConversationContext]:
        """Extract ConversationContext from various input formats.

        Priority: self.conversation_context (injected) > context dict
        """
        if self.conversation_context:
            return self.conversation_context
        if context and "conversation_context" in context:
            return context["conversation_context"]
        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process_user_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user request for emotional support with persona-driven dialogue.

        Flow:
        1. Get conversation context and record user turn
        2. Detect emotion and record in mood tracking
        3. Check if user has job-related needs → suggest WorkAgent handoff
        4. Generate conversational emotional support response
        5. Record agent response turn
        6. Return dialogue response with mood metadata

        Args:
            user_input: User's input text
            context: Additional context (emotional state, previous interactions, etc.)

        Returns:
            Dict with conversational response and metadata
        """
        self.logger.info(f"Processing emotion-related request: {user_input[:100]}...")

        # Step 1: Get conversation context and record user turn
        ctx = self._get_context(context)
        if ctx:
            ctx.add_turn("user", user_input)

        try:
            # Step 2: Detect emotion and record mood
            detected_emotion = self._detect_emotion(user_input, context)
            self.logger.info(f"Detected emotion category: {detected_emotion}")

            if ctx:
                ctx.record_mood(detected_emotion, trigger=user_input[:100])

            # Step 3: Check for job-related keywords → suggest handoff
            if self._detect_job_related(user_input):
                self.logger.info("Job-related content detected, offering WorkAgent handoff")
                response_text = await self._build_job_handoff_response(detected_emotion, ctx)
                if ctx:
                    ctx.add_turn("emotion_agent", response_text)
                return {
                    "agent": self.name,
                    "response": response_text,
                    "emotion_category": detected_emotion,
                    "suggest_handoff": "work_agent",
                    "status": "success"
                }

            # Step 4: Generate conversational emotional support
            # Try MCP emotion_support skill first
            use_skill = self._should_use_emotion_support_skill(detected_emotion, user_input, context)
            if use_skill and "advanced_emotion_support" in self.capabilities:
                try:
                    skill_response = await self._call_emotion_support_skill(user_input, context, detected_emotion)
                    response_text = await self._wrap_skill_response(skill_response, detected_emotion)
                except Exception as e:
                    self.logger.warning(f"emotion_support SKILL failed: {e}, using fallback")
                    response_text = await self._build_support_response(detected_emotion, user_input, ctx, context)
            else:
                # Build conversational support response using LLM
                response_text = await self._build_support_response(detected_emotion, user_input, ctx, context)

            # Step 5: Record agent response turn
            if ctx:
                ctx.add_turn("emotion_agent", response_text)

            return {
                "agent": self.name,
                "response": response_text,
                "emotion_category": detected_emotion,
                "status": "success"
            }

        except Exception as e:
            self.logger.error(f"Error processing emotion request: {e}")
            response_text = "抱歉，刚才出了点小问题。不过我在这里，如果你想聊聊的话，我随时都在。"
            return {
                "agent": self.name,
                "response": response_text,
                "status": "error",
                "error": str(e)
            }

    # ------------------------------------------------------------------
    # Emotion detection
    # ------------------------------------------------------------------

    def _detect_emotion(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Detect the primary emotion in user input.

        Args:
            user_input: User's input text
            context: Additional context

        Returns:
            Detected emotion category
        """
        input_lower = user_input.lower()

        # Check each emotion category
        for category, triggers in self.emotion_triggers.items():
            for trigger in triggers:
                if trigger in input_lower:
                    self.logger.debug(f"Detected emotion category '{category}' via trigger '{trigger}'")
                    return category

        # Check context for emotional state
        if context and "emotional_state" in context:
            emotional_state = context["emotional_state"].lower()
            for category in self.emotion_triggers.keys():
                if category in emotional_state:
                    return category

        # Check context for detected_emotion from BrainAgent
        if context and "detected_emotion" in context:
            emotion = context["detected_emotion"].lower()
            for category in self.emotion_triggers.keys():
                if category == emotion or category in emotion:
                    return category

        # Default
        return "general_encouragement"

    def _detect_job_related(self, user_input: str) -> bool:
        """Check if user input contains job-related keywords.

        Args:
            user_input: User's input text

        Returns:
            True if job-related keywords are detected
        """
        input_lower = user_input.lower()
        for keyword in _JOB_KEYWORDS:
            if keyword in input_lower:
                self.logger.debug(f"Detected job-related keyword: {keyword}")
                return True
        return False

    # ------------------------------------------------------------------
    # Response building
    # ------------------------------------------------------------------

    async def _build_support_response(self, emotion_category: str, user_input: str,
                                       ctx: Optional[ConversationContext] = None,
                                       context: Dict[str, Any] = None) -> str:
        """Build an emotional support response using LLM or fallback.

        Uses the task-specific emotion_agent_prompt (from emotion-recognition skill)
        for calibrated responses. Enriched with emotion analysis data (support_intensity,
        emotions, overall_state) from BrainAgent's demand_analysis.

        Args:
            emotion_category: Detected emotion category
            user_input: User's input text
            ctx: Conversation context (for mood trend awareness)
            context: Additional context dict (may contain demand_analysis from BrainAgent)

        Returns:
            Chinese dialogue text with emotional support
        """
        needs_intervention = False
        if ctx:
            trends = ctx.detect_mood_trends()
            needs_intervention = trends.get("needs_intervention", False)

        if self.llm_client:
            # Load task-specific system prompt (from emotion-recognition skill)
            # This contains behavioral guidelines for intensity calibration, crisis response, etc.
            emotion_prompt = _registry.get("system.emotion_agent_prompt.system_prompt", "")
            system_prompt = emotion_prompt if emotion_prompt else self.system_prompt

            # Build enriched context with emotion analysis data
            context_parts = [f"检测到情绪类别：{emotion_category}"]

            # Extract emotion analysis from context (passed from BrainAgent)
            demand_analysis = (context or {}).get("demand_analysis", {})
            emotions = demand_analysis.get("emotions", [])
            overall_state = demand_analysis.get("overall_state", "")
            support_intensity = demand_analysis.get("support_intensity", "")

            if emotions:
                emotion_labels = [e.get("label", "") for e in emotions if isinstance(e, dict)]
                context_parts.append(f"情绪分析：{', '.join(emotion_labels)}")
            if overall_state:
                context_parts.append(f"整体状态：{overall_state}")
            if support_intensity:
                context_parts.append(f"支持强度：{support_intensity}")

            if needs_intervention:
                context_parts.append("注意：用户近期情绪趋势需要更多关注和关怀")
            if ctx:
                history = ctx.get_formatted_history(max_turns=5)
                if history:
                    context_parts.append(f"近期对话：\n{history}")
                mood_trends = ctx.detect_mood_trends()
                if mood_trends:
                    context_parts.append(f"情绪趋势数据：{json.dumps(mood_trends, ensure_ascii=False)}")
            context_str = "\n\n".join(context_parts)

            return await self.llm_client.chat(
                system_prompt=system_prompt,
                user_input=user_input,
                context=context_str,
            )

        # Fallback (no LLM available)
        if needs_intervention and emotion_category in ("discouragement", "burnout_fatigue", "general_negative"):
            return (
                "我注意到你最近的心情一直不太好，我想让你知道，这完全没关系。"
                "求职过程本身就是起起伏伏的，没有人能一直保持积极。"
                "你已经做得很好了——你在这里，你在寻求帮助，你在努力，这本身就值得肯定。"
                "要不要先不聊求职的事，就随便聊聊？你今天有没有做什么让自己开心的事？"
            )

        return (
            "我在这里陪着你。不管你现在的感受如何，都是可以被理解的。"
            "如果你想聊聊发生了什么，我随时都在听。"
            "或者我们也可以随便聊聊别的，放松一下心情。你觉得怎么样？"
        )

    async def _build_job_handoff_response(self, emotion_category: str,
                                           ctx: Optional[ConversationContext] = None) -> str:
        """Build a response that acknowledges emotion and suggests WorkAgent handoff.

        Uses LLM when available, falls back to inline string.

        Args:
            emotion_category: Detected emotion category
            ctx: Conversation context

        Returns:
            Chinese dialogue text with handoff suggestion
        """
        if self.llm_client:
            context_parts = [f"用户当前情绪类别：{emotion_category}"]
            context_parts.append("用户提到了求职相关话题，需要转接到职业顾问。请在回应中先共情用户的感受，然后自然地将话题引导到职业顾问的帮助。")
            if ctx:
                history = ctx.get_formatted_history(max_turns=3)
                if history:
                    context_parts.append(f"近期对话：\n{history}")
            context_str = "\n\n".join(context_parts)

            return await self.llm_client.chat(
                system_prompt=self.system_prompt,
                user_input="",
                context=context_str,
            )

        # Fallback (no LLM available)
        ack = ""
        if emotion_category != "general_encouragement":
            ack = "我注意到你刚才提到了一些困扰你的感受，我完全理解。"

        return (
            f"{ack}听起来你也有一些具体的求职问题需要帮助。\n\n"
            f"我的主要角色是陪伴和支持你处理情绪方面的困扰。"
            f"关于具体的求职问题——比如简历优化、面试准备或者职业规划——"
            f"我建议请我的职业顾问同事来帮你，他会更专业地处理这些方面。\n\n"
            f"当然，如果你还是想先聊聊情绪方面的事，我也随时在这里。"
            f"你觉得怎么样？"
        )

    # ------------------------------------------------------------------
    # MCP skill integration
    # ------------------------------------------------------------------

    def _should_use_emotion_support_skill(self, emotion_category: str,
                                          user_input: str, context: Dict[str, Any] = None) -> bool:
        """Determine whether to use the emotion_support SKILL.

        Args:
            emotion_category: Detected emotion category
            user_input: User's input text
            context: Additional context

        Returns:
            True if emotion_support SKILL should be used
        """
        # Use skill for complex emotional situations
        complex_emotions = _registry.get("dialogues.support.complex_emotions", [])
        intensity_words = _registry.get("dialogues.support.intensity_words", [])

        if emotion_category in complex_emotions:
            input_lower = user_input.lower()
            if any(word in input_lower for word in intensity_words):
                return True

            if context and context.get("emotion_mentioned_count", 0) > 1:
                return True

        if "emotion_support" in user_input.lower() or "emotional support" in user_input.lower():
            return True

        return False

    async def _call_emotion_support_skill(self, user_input: str, context: Dict[str, Any],
                                          emotion_category: str) -> Dict[str, Any]:
        """Call the emotion_support SKILL via MCP Server.

        Args:
            user_input: User's input text
            context: Additional context
            emotion_category: Detected emotion category

        Returns:
            Response from emotion_support SKILL
        """
        request_data = {
            "skill_name": "emotion_support",
            "input_data": {
                "user_input": user_input,
                "emotion_category": emotion_category,
                "request_type": "emotional_support"
            },
            "context": context or {}
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.mcp_server_url}/execute",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info("emotion_support SKILL executed successfully")
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"MCP Server error: {response.status} - {error_text}")
                        raise RuntimeError(f"MCP Server returned {response.status}: {error_text}")
        except Exception as e:
            self.logger.error(f"Error calling emotion_support SKILL: {e}")
            raise

    async def _wrap_skill_response(self, skill_response: Dict[str, Any],
                                    emotion_category: str) -> str:
        """Wrap emotion_support SKILL response in conversational Chinese.

        Args:
            skill_response: Response from emotion_support SKILL
            emotion_category: Detected emotion category

        Returns:
            Chinese dialogue text with support content
        """
        support = skill_response.get("emotional_support", {})
        message = support.get("message", "")
        strategies = support.get("strategies", [])
        validation = support.get("validation", "")

        parts = []
        if validation:
            parts.append(validation)
        elif message:
            parts.append(message)
        else:
            fallback = await self._build_support_response(emotion_category, "", None)
            parts.append(fallback)

        if strategies:
            parts.append("\n这里有一些小建议可能对你有帮助：")
            for s in strategies[:3]:
                if isinstance(s, str):
                    parts.append(f"- {s}")
                elif isinstance(s, dict):
                    parts.append(f"- {s.get('suggestion', s.get('tip', ''))}")

        parts.append("\n我在这里陪着你，还有什么想聊的吗？")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def handle_text_message(self, message: AgentMessage) -> None:
        """Handle incoming text messages.

        Args:
            message: The message to handle
        """
        self.logger.info(f"Emotion Agent handling text message from {message.sender}")

        # Extract user input and context
        content = message.content
        user_input = content.get("text", "")
        context = content.get("context", {})

        # Track emotion mentions in context
        if "emotional_state" in content:
            context["emotional_state"] = content["emotional_state"]
            context["emotion_mentioned_count"] = context.get("emotion_mentioned_count", 0) + 1

        # Process the request
        response = await self.process_user_request(user_input, context)

        # Send response back to sender
        response_message = AgentMessage(
            sender=self.name,
            receiver=message.sender,
            content=response,
            message_type="response"
        )

        await self.send_message(response_message)
        self.logger.info(f"Emotion Agent response sent to {message.sender}")


# Factory function for easy instantiation
def create_emotion_agent(llm_client: Optional[LLMClient] = None,
                         mcp_server_url: str = "http://localhost:8000",
                         conversation_context: ConversationContext = None) -> EmotionAgent:
    """Create an EmotionAgent instance.

    Args:
        llm_client: Shared DeepSeek API client for LLM generation
        mcp_server_url: URL of the MCP Server
        conversation_context: Shared conversation state across agents

    Returns:
        EmotionAgent instance
    """
    return EmotionAgent(llm_client=llm_client,
                        mcp_server_url=mcp_server_url,
                        conversation_context=conversation_context)
