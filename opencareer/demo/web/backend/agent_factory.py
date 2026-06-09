"""Agent Factory - 创建和管理不同类型的 AI Agent

保持 LangChain 架构原样，提供适配器让现有后端能使用 CareerAgent。
"""

import logging
from typing import Optional, Any
from enum import Enum

from agent.simple_agent import SimpleAgent

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    SIMPLE = "simple"
    CAREER = "career"


class AgentFactory:
    """Agent 工厂类 - 创建不同类型的 Agent"""
    
    _instance: Optional['AgentFactory'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.logger = logging.getLogger(__name__)
            self._career_agent_instance = None
    
    def create_agent(self, agent_type: str = "simple", **kwargs):
        """
        创建指定类型的 Agent
        
        Args:
            agent_type: Agent 类型
            **kwargs: Agent 初始化参数
            
        Returns:
            Agent 实例
        """
        if agent_type == "simple":
            logger.info("Creating SimpleAgent")
            try:
                from llm.registry import get_llm_adapter
                from agent.simple_agent import SimpleAgent
                llm = get_llm_adapter()
                return SimpleAgent(llm)
            except Exception as e:
                logger.error(f"Failed to create SimpleAgent: {e}", exc_info=True)
                raise
        elif agent_type == "career":
            logger.info("Creating CareerAgent")
            try:
                # 导入完整的 CareerAgent
                from careers.agents.career_agent import CareerAgent
                
                # 从 kwargs 或环境获取参数
                api_key = kwargs.get("api_key")
                mcp_url = kwargs.get("mcp_url", "http://localhost:8001/mcp")
                memory_file = kwargs.get("memory_file", "career_memory.json")
                use_mcp = kwargs.get("use_mcp", True)
                
                return CareerAgent(
                    api_key=api_key,
                    mcp_url=mcp_url,
                    memory_file=memory_file,
                    use_mcp=use_mcp
                )
            except ImportError as e:
                logger.error(f"CareerAgent import failed (dependencies missing): {e}", exc_info=True)
                logger.warning("Falling back to SimpleAgent - install langchain dependencies for full functionality")
                return self.create_agent("simple")
            except Exception as e:
                logger.error(f"Failed to create CareerAgent: {e}", exc_info=True)
                logger.warning("Falling back to SimpleAgent")
                return self.create_agent("simple")
        else:
            logger.warning(f"Unknown agent type: {agent_type}, using SimpleAgent")
            return self.create_agent("simple")


# 全局工厂实例
_agent_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """获取全局 Agent 工厂实例"""
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = AgentFactory()
    return _agent_factory
