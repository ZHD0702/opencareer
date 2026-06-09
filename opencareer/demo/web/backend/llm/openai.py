from .base import LLMAdapter
import os
from dotenv import load_dotenv
from typing import AsyncGenerator, List, Dict
import json

load_dotenv()


class OpenAIAdapter(LLMAdapter):
    """OpenAI API 适配器"""
    
    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please configure it in .env file.")
    
    def _build_messages(self, messages: List[Dict[str, str]], system: str = None) -> List[Dict[str, str]]:
        """构建完整的消息列表，包含系统提示词"""
        result = []
        
        if system:
            result.append({"role": "system", "content": system})
        
        result.extend(messages)
        
        return result
    
    async def stream(self, messages: List[Dict[str, str]], system: str = None) -> AsyncGenerator[str, None]:
        """流式生成响应"""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            all_messages = self._build_messages(messages, system)
            
            async for chunk in await client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                stream=True
            ):
                content = chunk.choices[0].delta.content
                if content:
                    yield content
                    
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    async def invoke(self, messages: List[Dict[str, str]], system: str = None) -> str:
        """同步生成响应"""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            all_messages = self._build_messages(messages, system)
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=all_messages
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")