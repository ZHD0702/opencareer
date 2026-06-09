from .base import Agent
from typing import Any, Dict, List, Optional, AsyncGenerator
import json
import random

class SimpleAgent(Agent):
    """简单的 AI Agent - 基础版本"""
    
    def __init__(self, llm_adapter):
        super().__init__(llm_adapter)
        self.default_system = """你是 **OpenCareer** 的AI职业顾问助手。

## 核心角色
你是一位专业、温暖且富有同理心的职业发展顾问。

## 专业知识领域
- 简历优化
- 面试技巧
- 职业规划
- 薪资谈判
- 技能提升
- 职场沟通

## 回复格式要求（非常重要）
你必须以 JSON 数组的形式回复，每个元素是一个独立的短句。

### JSON 格式要求：
```json
[
  "你好呀！😊",
  "很高兴能帮助你。",
  "有什么我可以帮你的吗？"
]
```

### 具体要求：
1. 每句话在30-60个字之间
2. 句子自然完整，有足够的信息量
3. 适当使用表情符号（😊、💪、✨）
4. 每句话独立，不连在一起
5. 返回纯 JSON 数组，不要其他文字

## 示例回复

❌ 错误（太短）：
```json
[
  "你好呀！😊",
  "很高兴能帮助你。",
  "关于简历优化，我有建议。"
]
```

✅ 正确（自然长度）：
```json
[
  "你好呀！我是你的职业顾问，很高兴能帮助你！😊",
  "关于简历优化，我可以给你一些实用的建议和技巧。",
  "你想先聊聊简历的哪个方面呢？是内容还是排版？"
]
```

## 交流风格
1. 温暖有温度
2. 专业有深度
3. 简短有重点
4. 鼓励有方法

## 场景化指导
- 用户焦虑时：先表达理解
- 用户迷茫时：提问引导
- 用户成功时：真诚祝贺
- 用户失败时：提供建议

## 禁止行为
- 不说空话套话
- 不一次性输出太多
- 不用过于正式的语气
- 不要输出非 JSON 格式的内容

记住：只返回 JSON 数组，每个句子独立！"""
    
    async def execute(self, input_data: Any) -> Any:
        """实现抽象方法"""
        if isinstance(input_data, dict) and 'message' in input_data:
            messages = input_data.get('history', [])
            messages.append({'role': 'user', 'content': input_data['message']})
            return await self.chat(input_data.get('session_id', 'default'), messages)
        return None
    
    async def chat(self, session_id: str, messages: List[Dict[str, str]], system: str = None) -> AsyncGenerator[str, None]:
        """
        处理对话，返回流式响应
        """
        system_prompt = system or self.default_system
        
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        yield f"data: {json.dumps({'type': 'think_status', 'phase': 'thinking', 'status': 'AI正在思考...'})}\n\n"
        
        # 让LLM生成完整的JSON数组回复
        try:
            full_response = await self.llm.invoke(messages, system_prompt)
            sentences = self._parse_sentences(full_response)
            
            if not sentences:
                # 如果解析失败，尝试备用方案
                sentences = [full_response.strip()]
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            sentences = ["抱歉，我现在有点忙。", "稍后再试好吗？"]
        
        # 发送每个句子
        for i, sentence in enumerate(sentences):
            yield f"data: {json.dumps({
                'type': 'sentence',
                'content': sentence,
                'emotion': self._detect_emotion(sentence),
                'index': i,
                'is_last': i == len(sentences) - 1
            })}\n\n"
            
            # 延迟一下，模拟真人打字
            delay = self._calculate_delay(sentence)
            import asyncio
            await asyncio.sleep(delay / 1000)
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    def _parse_sentences(self, text: str) -> List[str]:
        """解析LLM返回的JSON数组"""
        try:
            # 尝试直接解析JSON
            result = json.loads(text)
            if isinstance(result, list):
                # 确保都是字符串
                return [str(s).strip() for s in result if str(s).strip()]
        except:
            pass
        
        # 尝试提取JSON部分
        try:
            # 查找第一个 [ 和最后一个 ]
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end+1]
                result = json.loads(json_str)
                if isinstance(result, list):
                    return [str(s).strip() for s in result if str(s).strip()]
        except:
            pass
        
        # 备用方案：用简单的标点拆分
        return self._fallback_split(text)
    
    def _fallback_split(self, text: str) -> List[str]:
        """备用拆分方法"""
        import re
        # 按主要标点拆分
        parts = re.split(r'[。？！\.！？]+', text)
        sentences = []
        
        for part in parts:
            part = part.strip()
            if part:
                # 如果太长，再按逗号拆分
                if len(part) > 20:
                    sub_parts = re.split(r'[，；,;]+', part)
                    for sub in sub_parts:
                        sub = sub.strip()
                        if sub:
                            sentences.append(sub)
                else:
                    sentences.append(part)
        
        return sentences[:5]  # 最多5句
    
    def _detect_emotion(self, sentence: str) -> str:
        """简单的情感检测"""
        empathy_keywords = ['理解', '明白', '心疼', '懂', '没关系', '我懂']
        excited_keywords = ['太棒', '太好了', '恭喜', '厉害', '加油', '好啊', '棒', '✨', '🎉']
        serious_keywords = ['注意', '建议', '必须', '一定', '重要', '关键', '记住', '要', '需要']
        
        if any(kw in sentence for kw in empathy_keywords):
            return 'empathy'
        if any(kw in sentence for kw in excited_keywords):
            return 'excited'
        if any(kw in sentence for kw in serious_keywords):
            return 'serious'
        return 'neutral'
    
    def _calculate_delay(self, sentence: str) -> int:
        """计算延迟时间"""
        length = len(sentence)
        base_delay = length * 80
        
        # 添加随机抖动
        jitter = 0.8 + random.random() * 0.4
        delay = base_delay * jitter
        
        return int(max(500, min(3000, delay)))
    
    async def think(self, user_message: str, history: List[Dict[str, str]] = None) -> Dict:
        """
        深度思考 - 分析用户意图和情绪
        """
        analysis_prompt = f"""分析以下用户消息，提取关键信息：

用户消息：{user_message}

请分析并返回 JSON 格式：
{{
    "intent": "意图分类：career_advice/skill_assessment/emotional_support/resume_help/job_search/casual_chat",
    "emotion": "情绪状态：positive/neutral/negative/anxious/excited",
    "keywords": ["关键词1", "关键词2"],
    "needs_support": true/false
}}
"""
        
        result = await self.llm.invoke(
            [{"role": "user", "content": analysis_prompt}],
            system="你是一个专业的意图分析助手，只返回 JSON 格式的分析结果。"
        )
        
        try:
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "intent": "casual_chat",
            "emotion": "neutral",
            "keywords": [],
            "needs_support": False
        }
    
    def get_streaming_prompt(self, intent: str = None, emotion: str = None) -> str:
        """
        获取针对特定意图和情绪的优化提示词
        """
        return self.default_system
