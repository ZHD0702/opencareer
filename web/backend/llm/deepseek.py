from .base import LLMAdapter
import os
from dotenv import load_dotenv
import httpx
from typing import AsyncGenerator, List, Dict
import json

load_dotenv()


class DeepSeekAdapter(LLMAdapter):
    """DeepSeek API 适配器"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.api_url = "https://api.deepseek.com/chat/completions"
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set. Please configure it in .env file.")
    
    def _build_messages(self, messages: List[Dict[str, str]], system: str = None) -> List[Dict[str, str]]:
        """构建完整的消息列表，包含系统提示词"""
        result = []
        
        if system:
            result.append({"role": "system", "content": system})
        
        result.extend(messages)
        
        return result
    
    async def stream(self, messages: List[Dict[str, str]], system: str = None) -> AsyncGenerator[str, None]:
        """流式生成响应"""
        all_messages = self._build_messages(messages, system)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                self.api_url,
                json={
                    "model": self.model,
                    "messages": all_messages,
                    "stream": True
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue

    async def invoke(self, messages: List[Dict[str, str]], system: str = None) -> str:
        """同步生成响应"""
        all_messages = self._build_messages(messages, system)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.api_url,
                json={
                    "model": self.model,
                    "messages": all_messages
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
