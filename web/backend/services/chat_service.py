from typing import AsyncGenerator, List, Dict
from dataclasses import dataclass
from utils.text_fragmenter import TextFragmenter

@dataclass
class ChatMessage:
    type: str
    content: str = ""
    phase: str = ""
    emotion: str = "neutral"
    index: int = 0
    is_last: bool = False

class ChatService:
    def __init__(self, llm_adapter):
        self.llm = llm_adapter
        self.fragmenter = TextFragmenter()
    
    async def stream_chat(self, session_id: str, user_message: str) -> AsyncGenerator[Dict, None]:
        yield ChatMessage(
            type="start",
            content="开始对话"
        ).__dict__
        
        yield ChatMessage(
            type="think_status",
            phase="analyzing",
            content="正在分析你的问题..."
        ).__dict__
        
        prompt = self._build_prompt(user_message)
        
        yield ChatMessage(
            type="think_status",
            phase="generating",
            content="正在生成回复..."
        ).__dict__
        
        response_chunks = []
        async for chunk in self.llm.stream(prompt):
            response_chunks.append(chunk)
        
        full_response = "".join(response_chunks)
        
        sentences = self.fragmenter.split(full_response)
        
        yield ChatMessage(
            type="think_complete",
            content="回复生成完成"
        ).__dict__
        
        for i, sentence in enumerate(sentences):
            yield ChatMessage(
                type="sentence",
                content=sentence.content,
                emotion=sentence.emotion,
                index=i,
                is_last=(i == len(sentences) - 1)
            ).__dict__
            
            if i < len(sentences) - 1:
                import asyncio
                await asyncio.sleep(sentence.delay_ms / 1000)
    
    def _build_prompt(self, user_message: str) -> str:
        return f"""你是求职顾问AI，正在和用户对话。

用户消息: {user_message}

请生成回复，要求：
1. 句子不要太长，每句话表达一个完整意思
2. 使用"。"、"？"、"！"等标点自然断句
3. 语气要自然、温暖、像真人聊天

直接输出回复内容，不需要其他说明。"""
