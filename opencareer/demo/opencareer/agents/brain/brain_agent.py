"""
Brain Agent for OpenCareer.

This is the entry point agent that analyzes user needs and hands off
to specialized agents (WorkAgent, EmotionAgent) for handling conversations.

Architecture: judge-and-handoff pattern.
BrainAgent does NOT orchestrate multi-agent responses. It judges the user's
demand type, records the context, and hands off to the appropriate single agent.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from ..base_agent import BaseAgent, AgentMessage, AgentRegistry
from ..llm_client import LLMClient
from ..conversation_context import ConversationContext
from ...extraction import extract_resume_fields
from ...extraction.resume_extractor import merge_extracted_fields
from ..persona import get_agent_persona
from opencareer.prompts.registry import get_global_registry

_registry = get_global_registry()


class BrainAgent(BaseAgent):
    """Brain Agent - Entry point and handoff controller for OpenCareer."""

    def __init__(self, llm_client: Optional[LLMClient] = None, deepseek_api_key: str = None,
                 agent_registry: AgentRegistry = None, conversation_context: ConversationContext = None):
        """Initialize Brain Agent.

        Args:
            llm_client: Shared LLM client for API calls
            deepseek_api_key: API key for DeepSeek (optional fallback for demo)
            agent_registry: Registry containing other agents
            conversation_context: Shared conversation state across agents
        """
        super().__init__(
            name="brain",
            description="Entry point agent that analyzes user needs and hands off to specialized agents"
        )

        # Capabilities reflect the judge-and-handoff pattern
        self.capabilities = [
            "demand_analysis",
            "agent_handoff",
            "llm_integration"
        ]

        # LLM client and fallback API key
        self.llm_client = llm_client
        self.deepseek_api_key = deepseek_api_key
        if not self.deepseek_api_key:
            self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

        self.agent_registry = agent_registry
        self.conversation_context = conversation_context

        # Cache persona system prompt for consistent dialogue
        self.system_prompt = get_agent_persona("brain")

        self.logger.info(f"Brain Agent initialized with LLM client: {'Yes' if self.llm_client else 'No'}, API key: {'Yes' if self.deepseek_api_key else 'No'}")

        # Register message handlers
        self.register_message_handler("user_request", self._handle_user_request)
        self.register_message_handler("agent_response", self._handle_agent_response)

        # Track pending requests
        self.pending_requests: Dict[str, Dict[str, Any]] = {}

        self.logger.info("Brain Agent initialized")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process_user_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze user demand and determine target agent for handoff.

        Judge-and-handoff pattern:
        1. Analyze user demand (classify as job_related / emotional / mixed / unknown)
        2. Record the conversation turn in shared context
        3. Determine single target agent (or None if clarification needed)
        4. Build dialogue response using persona
        5. Record handoff in ConversationContext
        6. Log in background (fire-and-forget)

        Args:
            user_input: User's input text
            context: Additional context (may contain ConversationContext or legacy dict)

        Returns:
            Dict with handoff instruction and dialogue response:
            - target_agent: "work_agent" / "emotion_agent" / None (clarify first)
            - response: Chinese dialogue text using persona
            - demand_analysis: Full analysis for downstream agents
            - agent, demand_type, confidence, status
        """
        self.logger.info(f"Processing user request: {user_input[:100]}")

        try:
            # Step 1: Analyze user demand
            demand_analysis = await self._analyze_demand(user_input, context)

            # Step 2: Record conversation turn if we have context
            ctx = self._get_context(context)
            if ctx:
                ctx.add_turn("user", user_input)

            # Step 2b: Extract user profile information from input
            await self._extract_user_profile(user_input, ctx)

            # Step 2c: Record emotion analysis for accumulated state tracking
            self._record_emotion_to_context(demand_analysis, ctx)

            # Step 3: Determine single target agent
            target_agent = self._determine_target_agent(demand_analysis)

            # Step 4: Build dialogue response using persona
            response_text = await self._build_dialogue_response(demand_analysis, target_agent, user_input, ctx)

            # Step 5: Record handoff if target identified
            if target_agent and ctx:
                reason = f"需求类型: {demand_analysis.get('demand_type', 'unknown')}"
                ctx.switch_agent(target_agent, reason)
                ctx.add_turn("brain", response_text)

            # Step 6: Log in background (fire-and-forget)
            await self._call_log_agent_async(user_input, demand_analysis, context)

            return {
                "agent": "brain",
                "target_agent": target_agent,
                "demand_type": demand_analysis.get("demand_type", "unknown"),
                "confidence": demand_analysis.get("confidence", 0.0),
                "demand_analysis": demand_analysis,
                "response": response_text,
                "status": "success"
            }

        except Exception as e:
            self.logger.error(f"Error processing user request: {e}")
            return {
                "agent": "brain",
                "target_agent": None,
                "error": str(e),
                "response": "抱歉，我在处理你的请求时遇到了问题。可以请你再说一遍吗？",
                "status": "error"
            }

    # ------------------------------------------------------------------
    # Opening greeting — called before the user's first input
    # ------------------------------------------------------------------

    async def get_opening_greeting(self) -> str:
        """Generate a warm opening greeting using the persona + LLM.

        The greeting is agentive (主动开场), setting a friendly tone and
        inviting the user to share their career or emotional needs.

        Returns:
            Greeting text in Chinese
        """
        if self.llm_client:
            try:
                greeting_prompt = (
                    "你刚刚上线，正在等待一位求职者向你咨询。"
                    "请用一段温暖、友好、开放的话主动打招呼，"
                    "介绍你是OpenCareer AI求职助手，"
                    "并邀请用户分享他们正在面临的求职困惑或情绪困扰。"
                    "要自然亲切，不要太长（100字以内）。"
                )
                greeting = await self.llm_client.chat(
                    system_prompt=self.system_prompt,
                    user_input=greeting_prompt,
                )
                if greeting and greeting.strip():
                    return greeting.strip()
            except Exception as e:
                self.logger.warning(f"LLM greeting failed, using fallback: {e}")

        # Fallback greeting
        return "你好！我是OpenCareer的AI求职助手，可以帮你解决求职问题或陪你聊聊心情。有什么我可以帮你的吗？"

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
    # Target agent determination (handoff logic)
    # ------------------------------------------------------------------

    def _determine_target_agent(self, demand_analysis: Dict[str, Any]) -> Optional[str]:
        """Determine which single agent should handle the conversation.

        Priority-based routing using emotion recognition data (extracted from
        emotion-recognition USAGE.md routing rules):

        1. Crisis state → EmotionAgent (highest priority, safety first)
        2. Force support (support_intensity == high) → EmotionAgent
        3. suggested_action == emotional_support → EmotionAgent
        4. demand_type == job_related → WorkAgent
        5. demand_type == mixed → EmotionAgent first (can handoff to WorkAgent)
        6. Low confidence / unknown → None (clarify with user)

        Args:
            demand_analysis: Result from demand analysis (includes emotion fields)

        Returns:
            Target agent name, or None if clarification is needed from user first
        """
        # Priority 1 & 2: Crisis or high-intensity emotional state
        overall_state = demand_analysis.get("overall_state", "")
        support_intensity = demand_analysis.get("support_intensity", "")

        if overall_state == "crisis" or support_intensity in ("high", "crisis"):
            return "emotion_agent"

        # Priority 3: suggested_action routing
        suggested_action = demand_analysis.get("suggested_action", "")
        if suggested_action == "emotional_support":
            return "emotion_agent"

        # Priority 4-6: Legacy demand_type routing
        demand_type = demand_analysis.get("demand_type", "unknown")
        confidence = demand_analysis.get("confidence", 0.0)

        if confidence < 0.3:
            return None

        if demand_type == "job_related":
            # Check accumulated emotional state across conversation turns.
            # If the user has repeatedly shown negative emotions, route to
            # emotion_agent first even if this turn appears work-related.
            # This implements: "当用户多次提及自己的情绪是负面状态之后，
            # 想要进行工作时，系统得首先根据之前对话得出的各情感置信度
            # 判断用户是否可以进行工作"
            if self.conversation_context:
                accumulated = self.conversation_context.get_accumulated_emotion_assessment()
                if accumulated.get("should_defer_work"):
                    self.logger.info(
                        f"Accumulated emotion override: {accumulated.get('reason')}"
                    )
                    demand_analysis["routing_override"] = "accumulated_emotion"
                    demand_analysis["accumulated_reason"] = accumulated.get("reason")
                    demand_analysis["accumulated_assessment"] = accumulated
                    return "emotion_agent"
            return "work_agent"
        elif demand_type in ("emotional", "mixed"):
            return "emotion_agent"
        else:  # unknown
            return None

    def _should_force_support(self, demand_analysis: Dict[str, Any]) -> bool:
        """Check if emotional support must be forced regardless of user's work intent.

        From emotion-recognition USAGE.md:
        Returns True when support_intensity is "high" or "crisis",
        or overall_state is "crisis".

        When True, the system must route to EmotionAgent even if the user
        explicitly asks for job-seeking assistance.

        Args:
            demand_analysis: Result from demand analysis

        Returns:
            True if emotional support should take priority over work tasks
        """
        support_intensity = demand_analysis.get("support_intensity", "none")
        overall_state = demand_analysis.get("overall_state", "")
        return support_intensity in ("high", "crisis") or overall_state == "crisis"

    async def _build_dialogue_response(self, demand_analysis: Dict[str, Any], target_agent: Optional[str],
                                       user_input: str, ctx: Optional[ConversationContext]) -> str:
        """Build a dialogue response using LLM or inline fallback.

        Now enriched with emotion analysis context (from emotion-recognition skill)
        for more calibrated handoff messages. When force_support is active, the
        response acknowledges emotional state before routing.

        Args:
            demand_analysis: Result from demand analysis (may include emotion fields)
            target_agent: Target agent name or None
            user_input: Original user input
            ctx: Conversation context (used to detect first turn)

        Returns:
            Chinese dialogue response text
        """
        demand_type = demand_analysis.get("demand_type", "unknown")
        is_first_turn = ctx is None or len([t for t in ctx.history if t.role == "user"]) <= 1
        force_support = self._should_force_support(demand_analysis)

        # Determine the handoff scenario (enriched with emotion context)
        if target_agent is None:
            scenario = "first_turn_greeting" if is_first_turn else "follow_up_clarification"
        elif target_agent == "work_agent":
            scenario = "handoff_to_work_agent"
        elif target_agent == "emotion_agent":
            if demand_analysis.get("routing_override") == "accumulated_emotion":
                scenario = "handoff_to_emotion_accumulated"
            elif force_support:
                scenario = "handoff_to_emotion_force"
            elif demand_type == "mixed":
                scenario = "handoff_to_emotion_mixed"
            else:
                scenario = "handoff_to_emotion_pure"
        else:
            scenario = "fallback_response"

        # Try LLM first
        if self.llm_client:
            try:
                history = ctx.get_formatted_history(max_turns=6) if ctx else ""

                # Build enriched context with emotion analysis data
                emotion_context = ""
                emotions = demand_analysis.get("emotions", [])
                if emotions:
                    emotion_labels = [e.get("label", "") for e in emotions if isinstance(e, dict)]
                    emotion_context = (
                        f"\n=== Emotion Analysis ===\n"
                        f"Emotions: {', '.join(emotion_labels)}\n"
                        f"Overall State: {demand_analysis.get('overall_state', 'neutral')}\n"
                        f"Support Intensity: {demand_analysis.get('support_intensity', 'none')}\n"
                    )
                # Add accumulated emotional state assessment for context-aware routing
                accumulated_context = ""
                if ctx and demand_analysis.get("routing_override") == "accumulated_emotion":
                    assessment = demand_analysis.get("accumulated_assessment", {})
                    if assessment:
                        accumulated_context = (
                            f"=== Accumulated Emotional State ===\n"
                            f"Assessment: {assessment.get('reason', '')}\n"
                            f"Negative Ratio: {assessment.get('negative_ratio', 0.0):.0%}\n"
                            f"Consecutive Negative Turns: {assessment.get('consecutive_negative', 0)}\n"
                            f"Action: 用户连续多轮表达负面情绪，本轮暂不处理工作需求，优先情感支持\n"
                        )

                profile_info = self._format_profile_for_context(ctx) if ctx else ""

                context_str = (
                    f"=== Demand Analysis ===\n"
                    f"Type: {demand_type}\n"
                    f"Confidence: {demand_analysis.get('confidence', 0.0)}\n"
                    f"Scenario: {scenario}\n"
                    f"Force Support: {force_support}\n"
                    f"Keywords: {demand_analysis.get('keywords', [])}\n"
                    f"{emotion_context}"
                    f"{accumulated_context}"
                    f"=== User Profile ===\n"
                    f"{profile_info}\n"
                    f"=== Conversation History ===\n"
                    f"{history}"
                )
                response = await self.llm_client.chat(
                    system_prompt=self.system_prompt,
                    user_input=user_input,
                    context=context_str,
                )
                if response and response.strip():
                    return response.strip()
            except Exception as e:
                self.logger.warning(f"LLM dialogue response failed, using fallback: {e}")

        # Fallback: hardcoded responses (emotion-aware)
        if force_support:
            return "我注意到你现在的情绪状态不太好，先让我们的情感支持伙伴来陪你聊聊，好吗？"

        fallbacks = {
            "handoff_to_work_agent": "我了解到你想寻求求职方面的帮助，马上为你转接到职业顾问。",
            "handoff_to_emotion_accumulated": "我注意到你这几轮对话中情绪状态不太好，先让我们的情感支持伙伴来陪你聊聊，好吗？",
            "handoff_to_emotion_mixed": "我理解你的感受，先请我们的情感支持伙伴来陪你聊聊。",
            "handoff_to_emotion_pure": "让我请我们的情感支持伙伴来陪你聊聊。",
        }
        if scenario in fallbacks:
            return fallbacks[scenario]

        if target_agent is None:
            if is_first_turn:
                return "你好！我是OpenCareer的AI求职助手，可以帮你解决求职问题或陪你聊聊心情。有什么我可以帮你的吗？"
            return "可以告诉我更多细节吗？你是想聊聊求职方面的事情，还是想说说心情？"

        return "让我看看怎么更好地帮助你……"

    # ------------------------------------------------------------------
    # Demand analysis (unchanged from original)
    # ------------------------------------------------------------------

    async def _analyze_demand(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze user demand using LLM with emotion recognition.

        Uses the rich emotion-analysis prompt (extracted from emotion-recognition
        skill) which outputs structured emotion data alongside demand classification.
        Falls back to simpler demand analysis if the rich prompt is unavailable,
        or to keyword-based simulation if no LLM client is configured.

        Args:
            user_input: User's input text
            context: Additional context

        Returns:
            Enriched demand analysis with emotion fields (emotions, overall_state,
            support_intensity) and legacy fields (demand_type, confidence, keywords)
        """
        self.logger.info("Analyzing user demand with emotion recognition")

        try:
            if self.llm_client:
                # Use the rich emotion analysis prompt (from emotion-recognition skill)
                system_prompt = _registry.get("system.emotion_analysis.system_prompt", "")
                if not system_prompt:
                    self.logger.warning(
                        "Emotion analysis prompt not found at 'system.emotion_analysis.system_prompt', "
                        "falling back to demand_analysis prompt"
                    )
                    system_prompt = _registry.get("system.demand_analysis.system_prompt", "")

                # Build context with conversation history (emotion-analysis prompt
                # recommends feeding the user's most recent 2-3 messages)
                ctx = self._get_context(context)
                history_str = ctx.get_formatted_history(max_turns=4) if ctx else ""
                context_parts = []
                if history_str:
                    context_parts.append(f"=== Recent Conversation History ===\n{history_str}")
                if context:
                    # Include non-history context (e.g. session metadata)
                    meta = {k: v for k, v in context.items() if k != "conversation_context"}
                    if meta:
                        context_parts.append(f"=== Additional Context ===\n{json.dumps(meta, ensure_ascii=False)}")
                context_parts.append(f"=== Current Input ===\n{user_input}")
                combined_context = "\n\n".join(context_parts)

                analysis = await self.llm_client.chat_json(
                    system_prompt=system_prompt,
                    user_input=user_input,
                    context=combined_context,
                )
                analysis = self._normalize_demand_analysis(analysis)
            else:
                analysis = self._simulate_demand_analysis(user_input, context)

            self.logger.debug(f"Demand analysis result: {analysis}")
            return analysis

        except Exception as e:
            self.logger.warning(f"Demand analysis failed, using fallback: {e}")
            return self._fallback_demand_analysis(user_input)

    def _normalize_demand_analysis(self, analysis: Any) -> Dict[str, Any]:
        """Validate and normalize demand analysis from LLM response.

        Handles both the rich emotion-analysis schema (from emotion-recognition skill)
        and the legacy demand_analysis schema. When the rich schema is detected, it
        derives demand_type from suggested_action/overall_state automatically.

        Rich schema fields preserved in output:
        - emotions: list of {label, confidence}
        - overall_state: positive | neutral | negative | crisis
        - suggested_action: work | emotional_support
        - support_intensity: none | low | medium | high | crisis
        - analysis_detail: {keywords_detected, tone_patterns, question_type, trend, ...}

        Args:
            analysis: Raw analysis from LLM (may be dict or other types)

        Returns:
            Normalized analysis with guaranteed valid fields + rich emotion fields
        """
        if not isinstance(analysis, dict):
            return {
                "demand_type": "unknown", "confidence": 0.0, "keywords": [],
                "explanation": "Invalid response format",
                "emotions": [], "overall_state": "neutral",
                "suggested_action": "work", "support_intensity": "none",
                "analysis_detail": {},
            }

        # ---- Detect which schema the LLM returned ----
        has_emotion_fields = "emotions" in analysis or "overall_state" in analysis or "suggested_action" in analysis

        if has_emotion_fields:
            return self._normalize_emotion_analysis(analysis)
        else:
            return self._normalize_legacy_analysis(analysis)

    def _normalize_emotion_analysis(self, analysis: dict) -> Dict[str, Any]:
        """Normalize the rich emotion-analysis schema and derive legacy fields."""

        # --- Emotion fields ---
        emotions = analysis.get("emotions", [])
        if not isinstance(emotions, list):
            emotions = []

        valid_states = ["positive", "neutral", "negative", "crisis"]
        overall_state = analysis.get("overall_state", "neutral")
        if overall_state not in valid_states:
            overall_state = "neutral"

        valid_actions = ["work", "emotional_support"]
        suggested_action = analysis.get("suggested_action", "work")
        if suggested_action not in valid_actions:
            suggested_action = "work"

        valid_intensities = ["none", "low", "medium", "high", "crisis"]
        support_intensity = analysis.get("support_intensity", "none")
        if support_intensity not in valid_intensities:
            support_intensity = "none"

        analysis_detail = analysis.get("analysis", {})
        if not isinstance(analysis_detail, dict):
            analysis_detail = {}

        # --- Derive demand_type from emotion fields ---
        if overall_state == "crisis":
            demand_type = "emotional"
        elif suggested_action == "emotional_support":
            demand_type = "emotional"
        elif suggested_action == "work":
            demand_type = "job_related"
        else:
            demand_type = "unknown"

        # Check for mixed: emotional suggestion but user also mentioned job keywords
        keywords_detected = analysis_detail.get("keywords_detected", [])
        if isinstance(keywords_detected, list):
            job_indicators = [
                "简历", "面试", "求职", "工作", "offer", "跳槽",
                "投递", "招聘", "resume", "interview", "job", "career",
            ]
            keyword_text = " ".join(str(k) for k in keywords_detected).lower()
            has_job_keywords = any(indicator.lower() in keyword_text for indicator in job_indicators)
            if demand_type == "emotional" and has_job_keywords:
                demand_type = "mixed"

        # --- Confidence ---
        confidence = 0.7  # default medium confidence for LLM analysis
        if emotions:
            confidences = [
                e.get("confidence", 0.0)
                for e in emotions
                if isinstance(e, dict) and isinstance(e.get("confidence"), (int, float))
            ]
            if confidences:
                confidence = sum(confidences) / len(confidences)
        confidence = max(0.0, min(1.0, confidence))

        # --- Keywords ---
        keywords = analysis_detail.get("keywords_detected", [])
        if not isinstance(keywords, list):
            keywords = []
        # Also add emotion labels as keywords for downstream use
        for e in emotions:
            if isinstance(e, dict) and e.get("label") and e["label"] not in keywords:
                keywords.append(e["label"])

        return {
            "demand_type": demand_type,
            "confidence": confidence,
            "keywords": keywords,
            "explanation": analysis_detail.get("reasoning", ""),
            # Rich emotion fields (for downstream agents)
            "emotions": emotions,
            "overall_state": overall_state,
            "suggested_action": suggested_action,
            "support_intensity": support_intensity,
            "analysis_detail": analysis_detail,
        }

    def _normalize_legacy_analysis(self, analysis: dict) -> Dict[str, Any]:
        """Normalize the legacy demand_analysis schema (backward compatibility)."""
        valid_types = ["job_related", "emotional", "mixed", "unknown"]
        demand_type = analysis.get("demand_type", "unknown")
        if demand_type not in valid_types:
            demand_type = "unknown"

        confidence = analysis.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        keywords = analysis.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []

        return {
            "demand_type": demand_type,
            "confidence": confidence,
            "keywords": keywords,
            "explanation": analysis.get("explanation", ""),
            # Default emotion fields
            "emotions": [],
            "overall_state": "neutral",
            "suggested_action": "work" if demand_type == "job_related" else "emotional_support",
            "support_intensity": "none",
            "analysis_detail": {},
        }

    def _simulate_demand_analysis(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simulate demand analysis for demo purposes.

        Args:
            user_input: User's input text
            context: Additional context

        Returns:
            Simulated demand analysis
        """
        input_lower = user_input.lower()

        job_keywords = _registry.get("system.demand_analysis.job_keywords", ["python", "skill", "learn", "study", "interview", "job", "career", "resume", "position", "apply", "application", "technical", "简历", "面试", "求职", "工作", "学习", "技能", "技术", "招聘", "跳槽", "投递", "职位", "实习", "校招", "社招", "offer", "找工作"])
        emotion_keywords = _registry.get("system.demand_analysis.emotion_keywords", ["stress", "stressed", "anxious", "anxiety", "nervous", "worried", "happy", "excited", "discouraged", "frustrated", "tired", "overwhelmed", "焦虑", "压力", "紧张", "沮丧", "难过", "迷茫", "烦躁", "崩溃", "失眠", "害怕", "担心", "不安", "痛苦", "抑郁", "急躁", "绝望", "心累"])

        job_count = sum(1 for word in job_keywords if word in input_lower)
        emotion_count = sum(1 for word in emotion_keywords if word in input_lower)

        if job_count > 0 and emotion_count > 0:
            return {
                "demand_type": "mixed",
                "confidence": min(0.9, 0.5 + (job_count + emotion_count) * 0.1),
                "keywords": [word for word in job_keywords + emotion_keywords if word in input_lower],
                "explanation": f"Mixed demand: found {job_count} job-related and {emotion_count} emotional keywords"
            }
        elif job_count > 0:
            return {
                "demand_type": "job_related",
                "confidence": min(0.95, 0.6 + job_count * 0.1),
                "keywords": [word for word in job_keywords if word in input_lower],
                "explanation": f"Job-related demand: found {job_count} job-related keywords"
            }
        elif emotion_count > 0:
            return {
                "demand_type": "emotional",
                "confidence": min(0.9, 0.5 + emotion_count * 0.1),
                "keywords": [word for word in emotion_keywords if word in input_lower],
                "explanation": f"Emotional demand: found {emotion_count} emotional keywords"
            }
        else:
            return {
                "demand_type": "unknown",
                "confidence": 0.3,
                "keywords": [],
                "explanation": "No clear job-related or emotional keywords detected"
            }

    def _fallback_demand_analysis(self, user_input: str) -> Dict[str, Any]:
        """Fallback demand analysis when API fails.

        Args:
            user_input: User's input text

        Returns:
            Basic demand analysis
        """
        return {
            "demand_type": "unknown",
            "confidence": 0.1,
            "keywords": [],
            "explanation": "Using fallback analysis due to API failure"
        }

    # ------------------------------------------------------------------
    # User profile extraction
    # ------------------------------------------------------------------

    async def _extract_user_profile(self, user_input: str, ctx: Optional[ConversationContext]) -> None:
        """Extract user profile information from user input using regex patterns.

        Detects common patterns for grade level, major, and school in Chinese text
        and updates ConversationContext.user_profile accordingly. Runs on every
        user message so that information can be accumulated over time.

        Args:
            user_input: The user's input text
            ctx: Conversation context to update
        """
        if not ctx:
            return

        profile = ctx.user_profile
        updated = False

        # --- Extract grade level (年级) ---
        # Patterns: "大一/大二/大三/大四/研一/研二/研三" + optional "学生"
        # e.g. "我是一名大二的软件工程学生", "我现在大四"
        import re
        grade_patterns = [
            (r'(大[一二三四])', 'grade_level'),
            (r'(研[一二三])', 'grade_level'),
            (r'(博[一二三])', 'grade_level'),
            (r'(高一|高二|高三)', 'grade_level'),
        ]
        for pattern, field in grade_patterns:
            match = re.search(pattern, user_input)
            if match:
                if profile.get(field) != match.group(1):
                    profile[field] = match.group(1)
                    updated = True
                break

        # --- Extract major (专业) ---
        # Patterns: "XX的学生", "XX专业", "学XX的"
        # Common majors that may appear before "学生" or "专业"
        major_patterns = [
            r'的(.{2,6})学生',           # "的软件工程学生" → group(1) = "软件工程"
            r'(.{2,6})专业',             # "软件工程专业" → group(1) = "软件工程"
            r'学(.{2,6})的',             # "学计算机的" → group(1) = "计算机"
        ]
        for pat in major_patterns:
            match = re.search(pat, user_input)
            if match:
                candidate = match.group(1).strip()
                # Filter out non-major patterns (common false positives)
                non_major = {"时候", "问题", "东西", "事情", "同学", "时间", "老师", "朋友"}
                if candidate and candidate not in non_major and len(candidate) <= 8:
                    if profile.get("major") != candidate:
                        profile["major"] = candidate
                        updated = True
                    break

        # --- Extract school (学校) ---
        # Patterns: "XX大学的学生", "在XX大学读书", "XX大学XX专业"
        school_match = re.search(r'([\u4e00-\u9fff]{2,8}(?:大学|学院))', user_input)
        if school_match:
            candidate = school_match.group(1)
            if profile.get("school") != candidate:
                profile["school"] = candidate
                updated = True

        # --- Update background_summary if anything changed ---
        if updated:
            parts = []
            if profile.get("grade_level"):
                parts.append(f"{profile['grade_level']}")
            if profile.get("major"):
                parts.append(f"{profile['major']}专业")
            if profile.get("school"):
                parts.append(profile["school"])
            if parts:
                summary = "".join(parts) + "学生"
                profile["background_summary"] = summary
            self.logger.info(f"User profile updated: grade={profile.get('grade_level')}, "
                             f"major={profile.get('major')}, school={profile.get('school')}")

        # --- Extract skill_focus from learning-related statements ---
        skill_patterns = [
            r'(?:学习|学|提升|掌握|练习|准备)\s*(.{2,20})',
        ]
        for pat in skill_patterns:
            match = re.search(pat, user_input)
            if match:
                skill = match.group(1).strip()
                # Avoid overly generic matches
                if skill and len(skill) <= 20 and skill not in profile.get("skill_focus", []):
                    current = list(profile.get("skill_focus", []))
                    current.append(skill)
                    profile["skill_focus"] = current
                    self.logger.info(f"Skill focus added: {skill}")

        # --- LLM-based extraction for full resume_data ---
        existing_resume = profile.get("resume_data")
        if existing_resume is not None and self.llm_client is not None:
            try:
                delta = await extract_resume_fields(
                    user_input, existing_resume, self.llm_client
                )
                if delta:
                    merge_extracted_fields(profile["resume_data"], delta)
                    self.logger.info(
                        f"LLM extraction merged {len(delta)} resume field(s): {list(delta.keys())}"
                    )
            except Exception as exc:
                self.logger.warning(f"LLM resume extraction failed (non-fatal): {exc}")

    def _format_profile_for_context(self, ctx: Optional[ConversationContext]) -> str:
        """Format user profile into a readable string for LLM context.

        Args:
            ctx: Conversation context

        Returns:
            Formatted string of known user info, or empty string if nothing known
        """
        if not ctx:
            return ""

        profile = ctx.user_profile
        parts = []

        background = profile.get("background_summary")
        if background:
            parts.append(f"已知信息：{background}")

        target_role = profile.get("target_role")
        if target_role:
            parts.append(f"目标岗位：{target_role}")

        skill_focus = profile.get("skill_focus", [])
        if skill_focus:
            parts.append(f"技能关注：{'、'.join(skill_focus[-5:])}")

        concerns = profile.get("common_concerns", [])
        if concerns:
            parts.append(f"关注话题：{'、'.join(concerns[-5:])}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Emotion recording for accumulated state tracking
    # ------------------------------------------------------------------

    def _record_emotion_to_context(self, demand_analysis: Dict[str, Any],
                                   ctx: Optional[ConversationContext]) -> None:
        """Record emotion analysis to context for accumulated state tracking.

        Stores the full emotion analysis in ctx.emotion_history and derives
        an English mood label for ctx.mood_history (used by detect_mood_trends).
        This enables BrainAgent to consider emotional trends across turns
        when routing decisions.

        Args:
            demand_analysis: Normalized analysis with emotion fields
            ctx: Conversation context to record to
        """
        if not ctx:
            return

        # Record full emotion analysis
        ctx.record_emotion_analysis(demand_analysis)

        # Derive mood label from top emotion for mood_history tracking
        emotions = demand_analysis.get("emotions", [])
        overall_state = demand_analysis.get("overall_state", "")

        if emotions and isinstance(emotions, list):
            top = emotions[0]
            if isinstance(top, dict) and top.get("label"):
                label = top["label"]
                # Map Chinese emotion labels to English mood categories
                mood_mapping = {
                    "焦虑": "anxious", "紧张": "anxious", "不安": "anxious",
                    "沮丧": "discouraged", "难过": "discouraged", "失落": "discouraged",
                    "迷茫": "discouraged", "绝望": "discouraged",
                    "压力": "stressed", "烦躁": "stressed", "崩溃": "stressed",
                    "害怕": "anxious", "担心": "anxious", "痛苦": "stressed",
                    "开心": "happy", "兴奋": "happy", "期待": "happy",
                    "平静": "neutral",
                }
                mood = mood_mapping.get(
                    label,
                    "neutral" if overall_state in ("positive", "neutral") else "stressed",
                )
                ctx.record_mood(mood, trigger=f"情感分析: {label}")
        elif overall_state == "crisis":
            ctx.record_mood("stressed", trigger="crisis状态")
        elif overall_state == "positive":
            ctx.record_mood("happy", trigger="积极状态")

    # ------------------------------------------------------------------
    # Background logging (fire-and-forget)
    # ------------------------------------------------------------------

    async def _call_log_agent_async(self, user_input: str, demand_analysis: Dict[str, Any],
                                   context: Dict[str, Any] = None) -> None:
        """Call Log Agent asynchronously for background logging.

        Args:
            user_input: User's input text
            demand_analysis: Demand analysis results
            context: Additional context
        """
        try:
            if self.agent_registry:
                log_agent = self.agent_registry.get_agent("log_agent")
                if log_agent:
                    log_data = {
                        "user_input": user_input,
                        "demand_analysis": demand_analysis,
                        "timestamp": asyncio.get_event_loop().time(),
                        "context": context or {}
                    }

                    asyncio.create_task(
                        log_agent.process_user_request(json.dumps(log_data), {"log_type": "user_interaction"})
                    )
            else:
                self.logger.debug("Log Agent called (simulated)")

        except Exception as e:
            self.logger.error(f"Error calling Log Agent: {e}")

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _handle_user_request(self, message: AgentMessage) -> None:
        """Handle user request messages.

        Args:
            message: User request message
        """
        user_input = message.content.get("text", "")
        context = message.content.get("context", {})

        self.logger.info(f"Handling user request via message: {user_input[:50]}...")

        response = await self.process_user_request(user_input, context)

        response_message = AgentMessage(
            sender=self.name,
            receiver=message.sender,
            content=response,
            message_type="user_response"
        )

        self.logger.info(f"Generated response for user request")

    async def _handle_agent_response(self, message: AgentMessage) -> None:
        """Handle responses from other agents.

        In the handoff pattern, this receives confirmation or error
        messages from agents that were handed off to.

        Args:
            message: Agent response message
        """
        self.logger.info(f"Received response from {message.sender}")
        self.logger.debug(f"Response content: {message.content}")

    async def background_work(self) -> None:
        """Perform background work for Brain Agent."""
        await asyncio.sleep(1)  # Placeholder

    async def handle_text_message(self, message: AgentMessage) -> None:
        """Handle direct text messages by routing through process_user_request.

        This allows BrainAgent to function as the conversational entry point
        when receiving unstructured text from users.

        Args:
            message: Text message from user
        """
        # Accept content as either string or dict
        if isinstance(message.content, str):
            user_input = message.content
            context = {}
        elif isinstance(message.content, dict):
            user_input = message.content.get("text", "")
            context = message.content.get("context", {})
        else:
            user_input = str(message.content)
            context = {}

        self.logger.info(f"Handling text message: {user_input[:50]}...")
        response = await self.process_user_request(user_input, context)

        # Send response back to the original sender
        response_message = AgentMessage(
            sender=self.name,
            receiver=message.sender,
            content=response,
            message_type="user_response"
        )
        self.logger.info(f"Generated response for text message")


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------

def create_brain_agent(llm_client: Optional[LLMClient] = None, deepseek_api_key: str = None,
                       agent_registry: AgentRegistry = None,
                       conversation_context: ConversationContext = None) -> BrainAgent:
    """Create a BrainAgent instance.

    Args:
        llm_client: Shared LLM client for API calls
        deepseek_api_key: API key for DeepSeek (fallback)
        agent_registry: Registry containing other agents
        conversation_context: Shared conversation state across agents

    Returns:
        BrainAgent instance
    """
    return BrainAgent(
        llm_client=llm_client,
        deepseek_api_key=deepseek_api_key,
        agent_registry=agent_registry,
        conversation_context=conversation_context
    )
