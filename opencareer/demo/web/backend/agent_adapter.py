"""Agent Adapter - 适配器统一不同 Agent 的接口"""

import logging
from typing import AsyncGenerator, List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class AgentAdapter:
    """Agent 适配器 - 统一 Agent 接口"""
    
    def __init__(self, agent):
        self.agent = agent
    
    async def chat(
        self,
        session_id: str,
        messages: List[Dict[str, str]],
        system: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        聊天接口 - 兼容不同类型的 Agent
        
        Args:
            session_id: 会话 ID
            messages: 消息历史
            system: 系统提示词（可选）
            
        Yields:
            SSE 格式的事件
        """
        # SimpleAgent 接口
        if hasattr(self.agent, 'chat') and callable(getattr(self.agent, 'chat')):
            try:
                async for event in self.agent.chat(session_id, messages, system):
                    yield event
            except Exception as e:
                logger.error(f"Agent chat failed: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        else:
            raise NotImplementedError("Agent does not support chat interface")
    
    async def execute(self, input_data):
        """Execute 接口"""
        if hasattr(self.agent, 'execute') and callable(getattr(self.agent, 'execute')):
            return await self.agent.execute(input_data)
        raise NotImplementedError("Agent does not support execute interface")
