from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, AsyncGenerator

class BaseAgent(ABC):
    """Agent 基类"""
    
    def __init__(self, llm_adapter=None):
        self.llm = llm_adapter
    
    @abstractmethod
    async def execute(self, input_data: Any) -> Any:
        pass

Agent = BaseAgent


class RouterAgent(BaseAgent):
    pass

class DialogueAgent(BaseAgent):
    pass

class AnalyzerAgent(BaseAgent):
    pass

class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {}
    
    def register(self, name: str, agent: BaseAgent):
        self.agents[name] = agent
    
    async def execute(self, input_data: Any, agent_name: str = None):
        if agent_name and agent_name in self.agents:
            return await self.agents[agent_name].execute(input_data)
        raise NotImplementedError("MultiAgent mode not implemented yet")
