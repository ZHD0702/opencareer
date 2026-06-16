from .base import LLMAdapter
import os
from dotenv import load_dotenv
from typing import AsyncGenerator, List, Dict

load_dotenv()


class ClaudeAdapter(LLMAdapter):
    """Anthropic Claude API 适配器"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Please configure it in .env file.")
    
    def _build_messages(self, messages: List[Dict[str, str]], system: str = None) -> List[Dict[str, str]]:
        """构建完整的消息列表"""
        result = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Claude 使用 user/assistant 角色
            if role == "system":
                continue
            elif role == "assistant":
                result.append({"role": "assistant", "content": content})
            else:
                result.append({"role": "user", "content": content})
        
        return result, system
    
    async def stream(self, messages: List[Dict[str, str]], system: str = None) -> AsyncGenerator[str, None]:
        """流式生成响应"""
        try:
            from anthropic import AsyncAnthropic
            
            client = AsyncAnthropic(api_key=self.api_key)
            
            all_messages, system_prompt = self._build_messages(messages, system)
            
            async with client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=system_prompt or "",
                messages=all_messages
            ) as stream:
                async for chunk in stream:
                    if chunk.type == "content_block_delta":
                        yield chunk.delta.text
                    
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Claude API error: {str(e)}")

    async def invoke(self, messages: List[Dict[str, str]], system: str = None) -> str:
        """同步生成响应"""
        try:
            from anthropic import AsyncAnthropic
            
            client = AsyncAnthropic(api_key=self.api_key)
            
            all_messages, system_prompt = self._build_messages(messages, system)
            
            response = await client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt or "",
                messages=all_messages
            )
            
            return response.content[0].text
            
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Claude API error: {str(e)}")