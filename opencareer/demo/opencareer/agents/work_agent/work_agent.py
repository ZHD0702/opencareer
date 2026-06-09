"""
Work Agent for OpenCareer system.

This agent handles career-related requests by routing to appropriate SKILLs
through the MCP Server. It uses the shared persona system for consistent
dialogue style and ConversationContext for multi-turn conversation coherence.
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


# Emotional distress keywords for brief support before handoff suggestion
_EMOTIONAL_DISTRESS_KEYWORDS: List[str] = _registry.get("dialogues.career_advice.emotional_distress_keywords", [])


class WorkAgent(BaseAgent):
    """Work Agent responsible for career-related assistance.

    Conversational agent that:
    - Engages in persona-driven Chinese dialogue for career help
    - Routes specialized requests to SKILLs (learning_plan, mock_interview)
    - Detects emotional distress and offers EmotionAgent handoff
    - Records all turns in shared ConversationContext for coherence
    """

    def __init__(self, llm_client: Optional[LLMClient] = None,
                 mcp_server_url: str = "http://localhost:8000",
                 conversation_context: ConversationContext = None):
        """Initialize Work Agent.

        Args:
            llm_client: Shared DeepSeek API client for LLM generation
            mcp_server_url: URL of the MCP Server
            conversation_context: Shared conversation state across agents
        """
        super().__init__(
            name="work_agent",
            description="Career expert that helps with resumes, interviews, skills, and career planning"
        )

        self.llm_client = llm_client
        self.mcp_server_url = mcp_server_url
        self.conversation_context = conversation_context
        self.last_skill_key: Optional[str] = None  # for multi-turn continuity

        # Available skills with Chinese keyword triggers (loaded from YAML)
        _triggers = _registry.get("dialogues.career_advice.skill_triggers", {})
        self.available_skills = {
            "learning_plan": {
                "name": "learning_plan",
                "description": "创建个性化学习计划，帮助用户系统性地学习和提升技能",
                "triggers": _triggers.get("learning_plan", []),
            },
            "mock_interview": {
                "name": "mock_interview",
                "description": "进行模拟面试练习，帮助用户准备真实面试场景",
                "triggers": _triggers.get("mock_interview", []),
            },
            "resume_cn": {
                "name": "resume_cn_career",
                "description": "中文简历生成、优化与ATS诊断，帮助用户打造高匹配度简历",
                "triggers": _triggers.get("resume_cn", []),
            },
        }

        # Fast lookup: skill name string → skill_info dict (for LLM routing result)
        self._skill_by_name: Dict[str, Dict[str, Any]] = {
            info["name"]: info for info in self.available_skills.values()
        }
        # Reverse lookup: skill name string → available_skills key (for last_skill_key)
        self._skill_key_by_name: Dict[str, str] = {}
        for key, info in self.available_skills.items():
            self._skill_key_by_name[info["name"]] = key

        # Register capabilities
        self.capabilities = [
            "learning_plan_creation",
            "mock_interview_facilitation",
            "career_advice",
            "skill_routing"
        ]

        # Cache persona system prompt for consistent dialogue
        self.system_prompt = get_agent_persona("work_agent")

        self.logger = logging.getLogger("agent.work_agent")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def _initialize(self) -> None:
        """Initialize connection to MCP Server."""
        self.logger.info(f"Initializing Work Agent with MCP Server: {self.mcp_server_url}")

        # Check MCP Server availability
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.mcp_server_url}/health") as response:
                    if response.status == 200:
                        self.logger.info("MCP Server is available")
                    else:
                        self.logger.warning(f"MCP Server health check failed: {response.status}")
        except Exception as e:
            self.logger.warning(f"Cannot connect to MCP Server: {e}")

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
        """Process user request with persona-driven conversational dialogue.

        Flow:
        1. Get conversation context and record user turn
        2. Check for emotional distress → brief support + offer handoff
        3. Determine which SKILL to call based on user input + context
        4. Call SKILL via MCP or provide conversational career advice
        5. Record agent response turn
        6. Return dialogue response with any skill metadata

        Args:
            user_input: User's input text
            context: Additional context (user info, session, etc.)

        Returns:
            Dict with conversational response and metadata
        """
        self.logger.info(f"Processing work-related request: {user_input[:100]}...")

        # Step 1: Get conversation context and record user turn
        ctx = self._get_context(context)
        if ctx:
            ctx.add_turn("user", user_input)

        skill_to_call = None
        try:
            # Step 2: Check context for high-intensity emotion (from brain analysis)
            demand_analysis = (context or {}).get("demand_analysis", {})
            support_intensity = demand_analysis.get("support_intensity", "")
            if support_intensity in ("high", "crisis"):
                self.logger.info(f"Context support_intensity={support_intensity}, offering handoff to emotion_agent")
                response_text = await self._build_distress_response(user_input, ctx)
                if ctx:
                    ctx.add_turn("work_agent", response_text)
                return {
                    "agent": self.name,
                    "response": response_text,
                    "suggest_handoff": "emotion_agent",
                    "status": "success"
                }

            # Step 3: Check for emotional distress by keywords
            if self._detect_emotional_distress(user_input):
                self.logger.info("Emotional distress detected, offering support + handoff")
                response_text = await self._build_distress_response(user_input, ctx)
                if ctx:
                    ctx.add_turn("work_agent", response_text)
                return {
                    "agent": self.name,
                    "response": response_text,
                    "suggest_handoff": "emotion_agent",
                    "status": "success"
                }

            # Step 4: Determine which SKILL to call
            skill_to_call = await self._determine_skill_to_call(user_input, context)

            # Step 4a: No specific SKILL matched → conversational career advice
            if not skill_to_call:
                self.logger.info("No specific SKILL matched, providing conversational advice")
                response_text = await self._build_conversational_advice(user_input, context, ctx)
            else:
                # Step 4b: Call SKILL via MCP
                skill_name = skill_to_call["name"]
                self.logger.info(f"Routing to SKILL: {skill_name}")

                # Build structured input_data for resume skill (LLM extraction)
                skill_input_data = None
                if skill_name == "resume_cn_career":
                    skill_input_data = await self._build_resume_skill_input(user_input)

                skill_response = await self._call_skill_via_mcp(
                    skill_name, user_input, context, input_data=skill_input_data
                )
                response_text = self._build_skill_result(skill_name, skill_response, user_input)

                # If skill result is empty (no data), fall back to conversational advice
                if not response_text:
                    self.logger.info(f"SKILL {skill_name} returned empty result, falling back to conversational advice")
                    response_text = await self._build_conversational_advice(user_input, context, ctx)

            # Step 5: Record agent response turn
            if ctx:
                ctx.add_turn("work_agent", response_text)

            return {
                "agent": self.name,
                "response": response_text,
                "skill_used": skill_to_call["name"] if skill_to_call else None,
                "status": "success"
            }

        except Exception as e:
            self.logger.error(f"Error processing work request: {e}")
            response_text = self._build_fallback_response()
            if ctx:
                ctx.add_turn("work_agent", response_text)
            return {
                "agent": self.name,
                "response": response_text,
                "skill_used": skill_to_call["name"] if skill_to_call else None,
                "status": "error",
                "error": str(e)
            }

    # ------------------------------------------------------------------
    # Emotional distress detection
    # ------------------------------------------------------------------

    def _detect_emotional_distress(self, user_input: str) -> bool:
        """Check if user input contains emotional distress signals.

        Args:
            user_input: User's input text

        Returns:
            True if distress keywords are detected
        """
        for keyword in _EMOTIONAL_DISTRESS_KEYWORDS:
            if keyword in user_input:
                return True
        return False

    async def _build_distress_response(self, user_input: str,
                                        ctx: Optional[ConversationContext]) -> str:
        """Build an emotional support response with handoff suggestion.

        Uses LLM when available, falls back to inline string.

        Args:
            user_input: User's input text
            ctx: Conversation context for mood tracking

        Returns:
            Chinese dialogue response with support and handoff offer
        """
        if ctx:
            ctx.record_mood("stressed", trigger=user_input[:100])

        if self.llm_client:
            context_parts = ["用户表现出情感困扰，请在回应中先共情，然后主动提出可以转接到情感支持伙伴"]
            if ctx:
                history = ctx.get_formatted_history(max_turns=5)
                if history:
                    context_parts.append(f"近期对话：\n{history}")
            context_str = "\n\n".join(context_parts)

            return await self.llm_client.chat(
                system_prompt=self.system_prompt,
                user_input=user_input,
                context=context_str,
            )

        # Fallback (no LLM available)
        return "抱歉，AI服务暂时不可用，请稍后再试。"

    # ------------------------------------------------------------------
    # Skill determination
    # ------------------------------------------------------------------

    async def _determine_skill_to_call(self, user_input: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Determine which SKILL to call based on user input and context.

        Uses LLM-based intent classification when the LLM client is available
        (understands full conversation context, not just keyword matching).
        Falls back to scoring-based keyword matching when LLM is unavailable
        or the call fails.

        Args:
            user_input: User's input text
            context: Additional context

        Returns:
            SKILL information or None if no match
        """

        # --- LLM-based routing (primary) ---
        if self.llm_client:
            try:
                result = await self._llm_route_skill(user_input, context)
                if result is not None:
                    return result
                # LLM explicitly returned None → no skill needed (e.g. casual chat)
                self.logger.info("LLM routing: no skill matched, treating as conversational")
                return None
            except Exception as e:
                self.logger.warning(
                    f"LLM routing failed ({e}), falling back to keyword matching"
                )

        # --- Keyword-based fallback (existing logic) ---
        return self._keyword_route_skill(user_input, context)

    # ------------------------------------------------------------------
    # LLM routing helper
    # ------------------------------------------------------------------

    async def _llm_route_skill(self, user_input: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Use LLM to classify user intent and select the appropriate skill.

        Returns:
            Skill info dict if a skill is selected, None if no skill needed,
            or raises an exception on failure (caller falls back to keywords).
        """
        # Build skill list for the prompt
        skill_descriptions = "\n".join(
            f"- {info['name']}: {info['description']}"
            for info in self.available_skills.values()
        )

        routing_system_prompt = (
            "你是一个意图分类器。根据用户输入和对话上下文，判断用户当前最需要哪个技能。\n\n"
            f"可用技能：\n{skill_descriptions}\n\n"
            "规则：\n"
            "1. 如果用户表达的需求明确对应某个技能，返回该技能名称\n"
            "2. 如果用户同时提到多个需求，选择最核心或最先表达的那个\n"
            "3. 如果用户只是打招呼、闲聊、确认或感谢，返回 null\n"
            "4. 如果用户的表述模糊但结合上下文能推断出意图，返回最匹配的技能\n\n"
            "严格按以下JSON格式输出，不要输出其他任何内容：\n"
            '{"skill": "<技能名称或null>"}'
        )

        # Build context with conversation history and user profile
        ctx = self._get_context(context)
        context_parts: list = []

        if ctx:
            history = ctx.get_formatted_history(max_turns=5)
            if history:
                context_parts.append(f"=== 对话历史 ===\n{history}")

            profile = ctx.user_profile or {}
            profile_parts = []
            if profile.get("target_role"):
                profile_parts.append(f"目标岗位：{profile['target_role']}")
            if profile.get("skill_focus"):
                profile_parts.append(f"技能方向：{profile['skill_focus']}")
            if profile.get("background_summary"):
                profile_parts.append(f"背景概述：{profile['background_summary']}")
            if profile_parts:
                context_parts.append("=== 用户信息 ===\n" + "\n".join(profile_parts))

        if self.last_skill_key:
            context_parts.append(
                f"注意：上一轮使用的技能是 '{self.last_skill_key}'，"
                f"如果用户输入承接上文（如\"再加一个\"、\"继续\"），优先选择该技能。"
            )

        context_str = "\n\n".join(context_parts) if context_parts else None

        result = await self.llm_client.chat_json(
            system_prompt=routing_system_prompt,
            user_input=user_input,
            context=context_str,
            temperature=0.1,
            max_tokens=80,
        )

        skill_name = result.get("skill")

        # Normalize: treat empty string or "none" as null
        if not skill_name or str(skill_name).lower() in ("none", "null", ""):
            return None

        skill_name = str(skill_name).strip()
        self.logger.info(f"LLM routing: skill={skill_name}")

        # Look up the skill
        if skill_name in self._skill_by_name:
            self.last_skill_key = self._skill_key_by_name.get(skill_name)
            return self._skill_by_name[skill_name]

        # Unknown skill name — warn caller so it can fall back
        self.logger.warning(
            f"LLM returned unknown skill '{skill_name}', "
            f"known: {list(self._skill_by_name.keys())}"
        )
        raise ValueError(f"Unknown skill: {skill_name}")

    # ------------------------------------------------------------------
    # Keyword routing helper
    # ------------------------------------------------------------------

    def _keyword_route_skill(self, user_input: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Keyword-based fallback routing with scoring and continuity.

        Uses scoring-based matching: longer triggers get higher base scores,
        and same-skill continuity from the previous turn gets a bonus.
        """
        input_lower = user_input.lower()
        matches: list = []  # (score, priority, skill_name, skill_info, matched_trigger)

        # Per-skill tiebreaker priority — resume_cn ranks highest because
        # resume inputs often contain words ("技能", "提升") that also match
        # other skills' triggers.
        SKILL_PRIORITY = {
            "resume_cn": 3,
            "mock_interview": 2,
            "learning_plan": 1,
        }

        for skill_name, skill_info in self.available_skills.items():
            triggers = skill_info["triggers"]
            for trigger in triggers:
                if trigger in input_lower:
                    # Base score = trigger length (longer = more specific)
                    score = len(trigger)
                    # Continuity bonus: same skill as last turn
                    if skill_name == self.last_skill_key:
                        score += 100
                    priority = SKILL_PRIORITY.get(skill_name, 0)
                    matches.append((score, priority, skill_name, skill_info, trigger))
                    self.logger.debug(
                        f"Matched trigger '{trigger}' for SKILL {skill_name} "
                        f"(score={score}, priority={priority})"
                    )

        if not matches:
            # No trigger matched — try context hints first
            if context:
                if context.get("request_type") == "learning_plan":
                    return self.available_skills.get("learning_plan")
                elif context.get("request_type") == "interview_prep":
                    return self.available_skills.get("mock_interview")
            # Fallback: multi-turn continuity — user continuing same topic
            # without re-using trigger keywords (e.g. "再加一个项目经历")
            if self.last_skill_key:
                fallback = self.available_skills.get(self.last_skill_key)
                if fallback:
                    self.logger.info(
                        f"No triggers matched, falling back to last skill '{self.last_skill_key}'"
                    )
                    return fallback
            self.last_skill_key = None
            return None

        # Sort by (score, priority) descending — priority breaks ties when
        # trigger lengths are equal.
        matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_score, best_priority, best_name, best_info, best_trigger = matches[0]

        self.logger.info(
            f"Keyword routing: selected SKILL '{best_name}' via trigger '{best_trigger}' "
            f"(score={best_score}, priority={best_priority}, "
            f"candidates={[(m[2], m[4], m[0]) for m in matches]})"
        )
        self.last_skill_key = best_name
        return best_info

    # ------------------------------------------------------------------
    # MCP skill calling
    # ------------------------------------------------------------------

    async def _call_skill_via_mcp(self, skill_name: str, user_input: str,
                                  context: Dict[str, Any] = None,
                                  input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a SKILL through the MCP Server.

        Args:
            skill_name: Name of the SKILL to call
            user_input: User's input text
            context: Additional context
            input_data: Optional structured input_data to merge (for resume skill etc.)

        Returns:
            SKILL execution result
        """
        self.logger.info(f"Calling SKILL {skill_name} via MCP Server")

        merged_input: Dict[str, Any] = {
            "user_input": user_input,
            "timestamp": asyncio.get_event_loop().time()
        }
        if input_data:
            merged_input.update(input_data)

        request_data = {
            "skill_name": skill_name,
            "input_data": merged_input,
            "context": context or {}
        }

        if context and "user_id" in context:
            request_data["input_data"]["user_id"] = context["user_id"]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.mcp_server_url}/execute",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"SKILL {skill_name} executed successfully")
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"MCP Server error: {response.status} - {error_text}")
                        raise RuntimeError(f"MCP Server returned {response.status}: {error_text}")

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout calling SKILL {skill_name}")
            raise RuntimeError(f"Timeout calling SKILL {skill_name}")
        except Exception as e:
            self.logger.error(f"Error calling MCP Server: {e}")
            raise

    # ------------------------------------------------------------------
    # Resume structured input builder
    # ------------------------------------------------------------------

    async def _build_resume_skill_input(self, user_input: str) -> Dict[str, Any]:
        """Use LLM to extract structured resume fields from free-text user input.

        Merges existing resume_data from conversation_context with fields
        freshly extracted from the current user message. Returns a dict of
        top-level fields that ResumeCnCareerSkill.execute() expects.

        Args:
            user_input: User's raw free-text message

        Returns:
            Dict with structured fields: action, name, target_role, job_type,
            industry, city, experience_years, phone, email, education, summary,
            experiences, projects, skills, certifications
        """
        # Start with any accumulated resume_data from previous turns
        base: Dict[str, Any] = {}
        if self.conversation_context:
            try:
                ctx_dict = self.conversation_context.to_dict()
                stored = ctx_dict.get("user_profile", {}).get("resume_data", {})
                if stored and isinstance(stored, dict):
                    # Flatten resume_data top-level fields into base
                    for key in (
                        "name", "target_position", "target_industry",
                        "target_location", "work_experience_years",
                        "phone", "email", "personal_summary",
                        "education_list", "work_experience_list",
                        "project_experience_list", "skill_list",
                        "certificate_list"
                    ):
                        val = stored.get(key)
                        if val:
                            base[key] = val
            except Exception:
                pass

        # Use LLM to extract structured fields from current user_input
        extracted: Dict[str, Any] = {}
        if self.llm_client:
            try:
                existing_str = json.dumps(base, ensure_ascii=False, indent=2) if base else "{}"
                sys_prompt = (
                    "你是一个简历信息提取器。从用户的输入中提取简历相关字段。\n"
                    "只提取用户在本次消息中明确提供的信息，不要编造。\n"
                    "返回一个 JSON 对象，字段名使用英文 key（如下所示），值为中文内容。\n"
                    "没有提到的字段设置为 null 或省略。\n\n"
                    "字段说明：\n"
                    '- action: 固定为 "generate"\n'
                    '- name: 姓名\n'
                    '- target_role: 目标岗位/职位\n'
                    '- job_type: 求职类型（校招/社招/转行/双语）\n'
                    '- industry: 行业方向\n'
                    '- city: 求职城市\n'
                    '- experience_years: 工作年限或应届说明\n'
                    '- phone: 手机号\n'
                    '- email: 邮箱\n'
                    '- education: 教育背景（学校/专业/学历/时间）\n'
                    '- summary: 职业摘要或自我介绍\n'
                    '- experiences: 经历要点列表（字符串数组）\n'
                    '- projects: 项目要点列表（字符串数组）\n'
                    '- skills: 技能列表（字符串数组）\n'
                    '- certifications: 证书或补充信息列表（字符串数组）\n\n'
                    f"已有的简历数据（供参考，本次只需返回新增或更新的字段）：\n{existing_str}"
                )
                user_prompt = f"用户输入：\n{user_input}\n\n请提取简历字段，返回 JSON。"
                extracted = await self.llm_client.chat_json(
                    system_prompt=sys_prompt,
                    user_input=user_prompt,
                    temperature=0.1
                )
                if not isinstance(extracted, dict):
                    extracted = {}
                self.logger.info(
                    f"LLM extracted {len(extracted)} resume field(s): {list(extracted.keys())}"
                )
            except Exception as e:
                self.logger.warning(f"LLM resume extraction failed, using regex fallback: {e}")

        # Merge: base + extracted (extracted wins)
        merged: Dict[str, Any] = {}
        for key, val in base.items():
            if val is not None and val != "" and val != []:
                merged[key] = val
        for key, val in extracted.items():
            if val is not None and val != "" and val != []:
                merged[key] = val

        # Map internal field names to what ResumeCnCareerSkill.execute() expects
        result: Dict[str, Any] = {"action": merged.get("action", "generate")}
        _str_map = (
            ("name", "name"),
            ("target_role", "target_position"),
            ("target_role", "target_role"),
            ("job_type", "job_type"),
            ("industry", "industry"),
            ("industry", "target_industry"),
            ("city", "city"),
            ("city", "target_location"),
            ("experience_years", "experience_years"),
            ("experience_years", "work_experience_years"),
            ("phone", "phone"),
            ("email", "email"),
            ("education", "education"),
            ("summary", "summary"),
            ("summary", "personal_summary"),
        )
        for skill_key, merged_key in _str_map:
            val = merged.get(merged_key)
            if val and isinstance(val, str) and skill_key not in result:
                result[skill_key] = val

        _list_map = (
            ("experiences", "experiences"),
            ("experiences", "work_experience_list"),
            ("projects", "projects"),
            ("projects", "project_experience_list"),
            ("skills", "skills"),
            ("skills", "skill_list"),
            ("certifications", "certifications"),
            ("certifications", "certificate_list"),
        )
        for skill_key, merged_key in _list_map:
            val = merged.get(merged_key)
            if val:
                if isinstance(val, list):
                    arr = [str(v).strip() for v in val if v]
                    if arr:
                        existing = result.get(skill_key, [])
                        if isinstance(existing, list):
                            result[skill_key] = existing + arr
                        else:
                            result[skill_key] = arr
                elif isinstance(val, str) and val.strip():
                    if skill_key not in result:
                        result[skill_key] = [val.strip()]

        self.logger.info(f"Built resume skill input: {len(result)} field(s)")
        return result

    # ------------------------------------------------------------------
    # Response building — conversational Chinese wrappers
    # ------------------------------------------------------------------

    async def _build_conversational_advice(self, user_input: str, context: Dict[str, Any] = None,
                                            ctx: Optional[ConversationContext] = None) -> str:
        """Build a conversational Chinese career advice response using LLM.

        Uses keyword-based topic detection from YAML, then generates
        a natural response via LLM or inline fallback.

        Args:
            user_input: User's input text
            context: Additional context
            ctx: Conversation context (for user profile updates)

        Returns:
            Chinese dialogue text with career advice
        """
        input_lower = user_input.lower()
        input_combined = user_input

        # Load topic detection keywords from YAML
        topic_keywords = _registry.get("dialogues.career_advice.topic_detection_keywords", {})

        topics = []
        for topic, keywords in topic_keywords.items():
            eng_match = any(word in input_lower for word in keywords)
            cn_match = any(word in input_combined for word in keywords)
            if eng_match or cn_match:
                topics.append(topic)

        # Update user profile if context available
        if ctx:
            if topics:
                ctx.add_concern(", ".join(topics))
            # Extract target role if mentioned
            prefixes = _registry.get("dialogues.career_advice.target_role_prefixes", [])
            for prefix in prefixes:
                if prefix in user_input:
                    ctx.update_profile("target_role", f"mentioned in: {user_input[-80:]}")

        if self.llm_client:
            context_parts = []
            if topics:
                context_parts.append(f"检测到用户关心的求职话题：{', '.join(topics)}")
            else:
                context_parts.append("未检测到特定话题，请提供通用的职业发展引导")

            # Calibrate tone based on support_intensity from brain analysis
            demand_analysis = (context or {}).get("demand_analysis", {})
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

            if ctx:
                history = ctx.get_formatted_history(max_turns=5)
                if history:
                    context_parts.append(f"近期对话历史：\n{history}")

                # Include user profile info if available
                profile = ctx.user_profile
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

            return await self.llm_client.chat(
                system_prompt=self.system_prompt,
                user_input=user_input,
                context=context_str,
            )

        # Fallback (no LLM available)
        return "抱歉，AI服务暂时不可用，请稍后再试。"

    def _build_skill_result(self, skill_name: str, skill_response: Dict[str, Any],
                            user_input: str) -> str:
        """Wrap a SKILL result in conversational Chinese.

        Args:
            skill_name: Name of the SKILL that was executed
            skill_response: Raw response from the SKILL
            user_input: Original user input

        Returns:
            Chinese dialogue text presenting the SKILL result
        """
        if skill_name == "learning_plan":
            return self._wrap_learning_plan(skill_response)
        elif skill_name == "mock_interview":
            return self._wrap_mock_interview(skill_response)
        elif skill_name == "resume_cn_career":
            return self._wrap_resume_cn(skill_response)
        else:
            return json.dumps(skill_response, ensure_ascii=False, indent=2)[:500]

    def _wrap_learning_plan(self, skill_response: Dict[str, Any]) -> str:
        """Wrap learning plan SKILL response in conversational Chinese.

        Args:
            skill_response: Response from learning plan SKILL

        Returns:
            Chinese dialogue text with learning plan details
        """
        plan = skill_response.get("learning_plan", {})
        if not plan:
            return ""

        goal = plan.get("goal", "未指定目标")
        timeline = plan.get("timeline_weeks", 0)
        skill_gaps = skill_response.get("skill_gaps", [])
        weekly = plan.get("weekly_schedule", [])

        parts = [f"我为你定制了一份学习计划，目标是：**{goal}**"]

        if timeline:
            parts.append(f"建议周期：约 **{timeline}周**")

        if skill_gaps:
            parts.append("重点提升方向：" + "、".join(skill_gaps[:5]))

        if weekly:
            first = weekly[0]
            week_num = first.get("week", 1)
            topics = first.get("topics", [])
            if topics:
                parts.append(f"第{week_num}周学习内容：" + "、".join(topics))

        return "\n".join(parts)

    def _wrap_mock_interview(self, skill_response: Dict[str, Any]) -> str:
        """Wrap mock interview SKILL response in conversational Chinese.

        Args:
            skill_response: Response from mock interview SKILL

        Returns:
            Chinese dialogue text with interview feedback
        """
        session = skill_response.get("interview_session", {})
        if not session:
            return ""

        questions = session.get("questions_asked", 0)
        score = session.get("total_score", 0)
        feedback = session.get("feedback_summary", "")
        feedback_points = session.get("feedback_points", [])

        parts = ["模拟面试完成，这是你的表现总结："]

        if questions:
            parts.append(f"回答题目数：**{questions}道**")
        if score:
            parts.append(f"综合评分：**{score}/100**")
        if feedback:
            parts.append(f"总体反馈：{feedback}")

        if feedback_points:
            points_list = []
            for point in feedback_points[:3]:
                if isinstance(point, str):
                    points_list.append(f"- {point}")
                elif isinstance(point, dict):
                    points_list.append(f"- {point.get('suggestion', point.get('feedback', ''))}")
            if points_list:
                parts.append("**主要改进建议：**\n" + "\n".join(points_list))

        return "\n".join(parts)

    def _wrap_resume_cn(self, skill_response: Dict[str, Any]) -> str:
        """Wrap resume CN SKILL response in conversational Chinese.

        Handles the actual skill output format from ResumeCnCareerSkill.

        Args:
            skill_response: Response from resume CN SKILL

        Returns:
            Chinese dialogue text with resume guidance or diagnosis
        """
        if not skill_response.get("ok"):
            error_msg = skill_response.get("error", "未知错误")
            return f"简历处理遇到问题：{error_msg}"

        action = skill_response.get("action", "generate")
        resume = skill_response.get("resume", {})
        ats_report = skill_response.get("ats_report", {})
        notes = skill_response.get("notes", [])

        if not resume:
            return ""

        contact = resume.get("contact", {})
        name = contact.get("name", "未填写")
        title = contact.get("title", "")
        summary_text = resume.get("summary", "")
        sections = resume.get("sections", [])

        parts = ["# 简历生成报告"]

        # Basic info
        if title:
            parts.append(f"**{name}** — {title}")
        else:
            parts.append(f"**姓名：{name}**")

        if contact.get("location"):
            parts.append(f"城市：{contact['location']}")
        if contact.get("phone"):
            parts.append(f"电话：{contact['phone']}")
        if contact.get("email"):
            parts.append(f"邮箱：{contact['email']}")
        parts.append("")

        # Summary
        if summary_text:
            parts.append("**职业摘要：**")
            parts.append(summary_text)
            parts.append("")

        # Sections
        for section in sections:
            section_title = section.get("title", "")
            if section_title:
                parts.append(f"**{section_title}：**")
            for item in section.get("items", []):
                heading = item.get("heading", "")
                if heading:
                    parts.append(f"- {heading}")
                for bullet in item.get("bullets", []):
                    parts.append(f"  - {bullet}")
            parts.append("")

        # ATS report
        if ats_report:
            ats_score = ats_report.get("score", 0)
            ats_issues = ats_report.get("issues", [])
            ats_suggestions = ats_report.get("suggestions", [])
            coverage = ats_report.get("keyword_coverage_percent")

            parts.append("---")
            parts.append(f"**ATS 评分：{ats_score}/100**")
            if coverage is not None:
                parts.append(f"关键词覆盖率：{coverage}%")

            if ats_issues:
                parts.append("")
                parts.append("**发现的问题：**")
                for issue in ats_issues:
                    parts.append(f"- {issue}")

            if ats_suggestions:
                parts.append("")
                parts.append("**优化建议：**")
                for suggestion in ats_suggestions:
                    parts.append(f"- {suggestion}")

            parts.append("")

        # Notes
        if notes:
            parts.append("---")
            parts.append("**提示：**")
            for note in notes:
                parts.append(f"- {note}")

        return "\n".join(parts)

    def _build_fallback_response(self) -> str:
        """Build a conversational Chinese fallback when a SKILL call fails.

        Returns:
            Chinese dialogue text explaining the issue and offering alternatives
        """
        return "抱歉，AI服务暂时不可用，请稍后再试。"

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def handle_text_message(self, message: AgentMessage) -> None:
        """Handle incoming text messages.

        Args:
            message: The message to handle
        """
        self.logger.info(f"Work Agent handling text message from {message.sender}")

        content = message.content
        user_input = content.get("text", "")
        context = content.get("context", {})

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
        self.logger.info(f"Work Agent response sent to {message.sender}")


# Factory function for easy instantiation
def create_work_agent(llm_client: Optional[LLMClient] = None,
                      mcp_server_url: str = "http://localhost:8000",
                      conversation_context: ConversationContext = None) -> WorkAgent:
    """Create a WorkAgent instance.

    Args:
        llm_client: Shared DeepSeek API client for LLM generation
        mcp_server_url: URL of the MCP Server
        conversation_context: Shared conversation state across agents

    Returns:
        WorkAgent instance
    """
    return WorkAgent(llm_client=llm_client,
                     mcp_server_url=mcp_server_url,
                     conversation_context=conversation_context)
