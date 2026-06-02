from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict

class LLMAdapter(ABC):
    """LLM 适配器基类，定义统一的接口"""
    
    @abstractmethod
    async def stream(self, messages: List[Dict[str, str]], system: str = None) -> AsyncGenerator[str, None]:
        """
        流式生成响应
        
        Args:
            messages: 对话历史 [{"role": "user/assistant", "content": "..."}]
            system: 系统提示词（可选）
        
        Yields:
            生成的文本片段
        """
        pass

    @abstractmethod
    async def invoke(self, messages: List[Dict[str, str]], system: str = None) -> str:
        """
        同步生成响应
        
        Args:
            messages: 对话历史 [{"role": "user/assistant", "content": "..."}]
            system: 系统提示词（可选）
        
        Returns:
            生成的完整文本
        """
        pass
