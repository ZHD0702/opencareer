"""MultiAgent Orchestrator - 多 Agent 编排器

负责协调多个 Agent 之间的协作，根据用户请求选择最合适的 Agent 来处理。
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncIterator
from enum import Enum

# 添加项目根目录到 Python 路径
_project_root = Path(__file__).resolve().parent.parent.parent  # F:\opencareer\web\backend\agents\orchestrator.py -> F:\opencareer
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agent.base import BaseAgent
from agent.simple_agent import SimpleAgent
from opencareer.agents.career_agent import CareerAgent

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Agent 角色枚举"""
    GENERAL = "general"           # 通用对话
    CAREER_ADVISOR = "career"     # 职业顾问
    RESUME_EXPERT = "resume"      # 简历专家
    EMOTION_SUPPORT = "emotion"   # 情感支持
    SKILL_ASSESSOR = "skill"      # 技能评估


class Orchestrator:
    """MultiAgent 编排器
    
    负责：
    1. 根据用户意图选择最合适的 Agent
    2. 协调多个 Agent 之间的信息传递
    3. 处理复杂任务的多步执行
    4. 管理 Agent 间的状态共享
    """
    
    _instance: Optional['Orchestrator'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._agents: Dict[str, Any] = {}
            self._default_agent = None
    
    def _initialize_agents(self):
        """初始化所有可用的 Agent"""
        from llm.registry import get_llm_adapter
        
        if "general" not in self._agents:
            try:
                llm = get_llm_adapter()
                self._agents["general"] = SimpleAgent(llm)
                self._default_agent = self._agents["general"]
                logger.info("Initialized General Agent")
            except Exception as e:
                logger.warning(f"Failed to initialize General Agent: {e}")
        
        if "career" not in self._agents:
            try:
                self._agents["career"] = CareerAgent(use_mcp=True)
                logger.info("Initialized Career Agent")
            except Exception as e:
                logger.warning(f"Failed to initialize Career Agent: {e}")
    
    async def _classify_intent(self, user_input: str) -> AgentRole:
        """
        使用 LLM 对用户意图进行分类
        
        Returns:
            AgentRole: 最合适的 Agent 角色
        """
        from llm.registry import get_llm_adapter
        
        intent_patterns = {
            AgentRole.CAREER_ADVISOR: [
                "找工作", "求职", "面试", "薪资", "职业发展", "跳槽", "简历",
                "offer", "薪资谈判", "职业规划", "晋升", "职场", "工作",
            ],
            AgentRole.RESUME_EXPERT: [
                "简历", "CV", "优化简历", "简历模板", "简历修改",
                "ATS", "简历分析", "简历建议",
            ],
            AgentRole.EMOTION_SUPPORT: [
                "焦虑", "压力", "迷茫", "沮丧", "难过", "担心", "烦恼",
                "心情", "情绪", "心累", "疲惫",
            ],
            AgentRole.SKILL_ASSESSOR: [
                "技能", "学习", "掌握", "技术", "能力", "评估",
                "技能树", "学习路径", "技术栈",
            ],
        }
        
        # 简单的规则匹配
        user_input_lower = user_input.lower()
        
        for role, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in user_input_lower:
                    logger.info(f"Intent classified as {role} based on pattern: {pattern}")
                    return role
        
        return AgentRole.GENERAL
    
    async def _select_agent(self, intent: AgentRole) -> Any:
        """根据意图选择合适的 Agent"""
        agent_mapping = {
            AgentRole.GENERAL: "general",
            AgentRole.CAREER_ADVISOR: "career",
            AgentRole.RESUME_EXPERT: "career",
            AgentRole.EMOTION_SUPPORT: "career",
            AgentRole.SKILL_ASSESSOR: "career",
        }
        
        agent_key = agent_mapping.get(intent, "general")
        
        if agent_key in self._agents:
            return self._agents[agent_key]
        
        return self._default_agent
    
    async def route(self, user_input: str) -> AsyncIterator[str]:
        """
        路由用户请求到最合适的 Agent
        
        Args:
            user_input: 用户输入
            
        Yields:
            响应内容片段
        """
        self._initialize_agents()
        
        if not self._default_agent:
            yield "抱歉，系统暂不可用，请稍后重试。"
            return
        
        # 分类意图
        intent = await self._classify_intent(user_input)
        logger.info(f"Routing request with intent: {intent}")
        
        # 选择 Agent
        agent = await self._select_agent(intent)
        
        if agent is None:
            agent = self._default_agent
        
        # 执行请求
        try:
            if hasattr(agent, "stream_chat"):
                async for token in agent.stream_chat(user_input):
                    yield token
            elif hasattr(agent, "chat"):
                # 非流式响应
                response = await agent.chat(user_input) if hasattr(agent.chat, "__await__") else agent.chat(user_input)
                yield response
            else:
                yield "Agent 不支持对话功能"
                
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            # 降级到默认 Agent
            if agent != self._default_agent and self._default_agent:
                yield f"[切换到通用模式] "
                async for token in self._default_agent.chat("", [{"role": "user", "content": user_input}]):
                    if token.startswith('data: '):
                        import json
                        try:
                            data = json.loads(token[6:])
                            if data.get('type') == 'sentence':
                                yield data.get('content', '')
                        except:
                            pass
    
    async def delegate(self, user_input: str, agent_type: str) -> AsyncIterator[str]:
        """
        直接委托给指定类型的 Agent
        
        Args:
            user_input: 用户输入
            agent_type: Agent 类型 (general, career)
            
        Yields:
            响应内容片段
        """
        self._initialize_agents()
        
        if agent_type in self._agents:
            agent = self._agents[agent_type]
        else:
            agent = self._default_agent
            agent_type = "general"
        
        logger.info(f"Direct delegation to {agent_type} agent")
        
        try:
            if hasattr(agent, "stream_chat"):
                async for token in agent.stream_chat(user_input):
                    yield token
            elif hasattr(agent, "chat"):
                response = await agent.chat(user_input) if hasattr(agent.chat, "__await__") else agent.chat(user_input)
                yield response
            else:
                yield "Agent 不支持对话功能"
                
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            yield f"Agent 执行失败: {str(e)}"
    
    def get_available_agents(self) -> List[str]:
        """获取所有可用的 Agent 列表"""
        self._initialize_agents()
        return list(self._agents.keys())
    
    def get_agent_info(self, agent_type: str) -> Dict[str, Any]:
        """获取指定 Agent 的信息"""
        self._initialize_agents()
        
        agent_info = {
            "general": {
                "name": "通用对话 Agent",
                "description": "处理通用对话请求",
                "capabilities": ["通用问答", "信息查询", "日常对话"],
            },
            "career": {
                "name": "职业顾问 Agent",
                "description": "专业的职业发展助手",
                "capabilities": ["职业建议", "简历优化", "面试准备", "情感支持", "技能评估"],
            },
        }
        
        return agent_info.get(agent_type, {})


# 全局编排器实例
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """获取全局编排器实例"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator