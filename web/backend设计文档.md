# AI 职业伴侣 - 后端设计文档

**版本**: v1.0  
**日期**: 2026-05-25  
**状态**: 待开发

---

## 一、项目概述

### 1.1 项目背景

AI 职业伴侣是一个面向求职者的智能助手应用，前端已基于 React + TypeScript + TailwindCSS 实现。本文档定义后端架构设计，用于指导后续开发。

### 1.2 核心架构理念

```
┌─────────────────────────────────────────────────────────────┐
│                   Frontend (Talker)                          │
│            前端负责快速响应，即时交互                          │
│            接收后端指导，动态调整回答                          │
└────────────────────────────┬────────────────────────────────┘
                             │ SSE 流式通信
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend (Thinker)                          │
│            后端负责深度思考，策略指导                          │
│            意图分析 + 回答校正 + 情感检测                      │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 设计原则

| 原则 | 说明 |
|------|------|
| **渐进式开发** | 从 MVP 开始，逐步迭代到完整架构 |
| **LLM 可切换** | 当前使用 DeepSeek，预留切换接口 |
| **碎片化输出** | 模拟真人对话，句子级流式输出 |
| **可扩展 Agent** | 简单架构设计，方便后续接入多 Agent |

---

## 二、技术选型

### 2.1 技术栈

| 组件 | 选择 | 说明 |
|------|------|------|
| **语言** | Python 3.11+ | AI 生态丰富，异步支持好 |
| **框架** | FastAPI | SSE 原生支持，自动 OpenAPI |
| **LLM** | DeepSeek API (开发) | 预留切换接口至 GPT-4/Claude |
| **数据库** | SQLite (开发) → PostgreSQL (生产) | SQLAlchemy ORM 统一管理 |
| **异步任务** | asyncio | 轻量级后台任务处理 |
| **部署** | 本地运行 → Docker | 开发完成后容器化 |

### 2.2 依赖清单

```
# requirements.txt
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
sqlalchemy>=2.0.0
aiosqlite>=0.19.0
python-dotenv>=1.0.0
httpx>=0.27.0
pydantic>=2.0.0
```

---

## 三、系统架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Frontend (React)                            │
│                                                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│   │   ChatInput  │  │ MessageList  │  │   Sidebar    │             │
│   │   用户输入    │  │  对话展示    │  │  侧边栏功能  │             │
│   └──────────────┘  └──────────────┘  └──────────────┘             │
│                              │                                       │
│                              ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │                  useChatStream (Hook)                    │       │
│   │  • SSE 接收流式数据                                       │       │
│   │  • 碎片化气泡渲染                                         │       │
│   │  • 思考状态显示                                           │       │
│   └─────────────────────────────────────────────────────────┘       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ HTTP/SSE
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          FastAPI Backend                            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      API Routes                              │   │
│  │  POST /api/chat/{session_id}      SSE 流式对话               │   │
│  │  POST /api/sessions              创建会话                    │   │
│  │  GET  /api/emotion/trends/{id}   情绪趋势                    │   │
│  │  GET  /api/skill-assessment/{id} 技能评估                    │   │
│  │  GET  /api/resume/{id}           简历信息                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      ChatService                             │   │
│  │                                                              │   │
│  │  ┌────────────────┐    ┌────────────────┐                    │   │
│  │  │ TextFragmenter │───▶│ StreamSender  │                    │   │
│  │  │  句子拆分      │    │  流式发送      │                    │   │
│  │  └────────────────┘    └────────────────┘                    │   │
│  │                              │                               │   │
│  │  ┌───────────────────────────┴────────────────────────┐    │   │
│  │  │                  SimpleAgent                       │    │   │
│  │  │                                                      │    │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │    │   │
│  │  │  │  Intent   │  │ Emotion  │  │ Response  │        │    │   │
│  │  │  │  Analyzer │  │ Detector │  │ Generator │        │    │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘        │    │   │
│  │  │                                                      │    │   │
│  │  │         ↑ 可扩展为 MultiAgentOrchestrator           │    │   │
│  │  └──────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     LLM Adapter                             │   │
│  │                                                              │   │
│  │         ┌───────────┐                                        │   │
│  │         │  DeepSeek │  ←  当前配置                           │   │
│  │         └───────────┘                                        │   │
│  │         ┌───────────┐                                        │   │
│  │         │  OpenAI   │  ←  预留接口                           │   │
│  │         └───────────┘                                        │   │
│  │         ┌───────────┐                                        │   │
│  │         │  Claude   │  ←  预留接口                           │   │
│  │         └───────────┘                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     Database (SQLite)                        │   │
│  │                                                              │   │
│  │  sessions | messages | emotion_records | skill_records     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 模块职责

| 模块 | 职责 | 备注 |
|------|------|------|
| **API Routes** | HTTP/SSE 端点，接收请求，返回响应 | FastAPI 路由 |
| **ChatService** | 对话流程控制，句子拆分，流式发送 | 核心服务 |
| **SimpleAgent** | 简单 Agent，处理对话逻辑 | Phase 1 实现 |
| **LLM Adapter** | LLM 调用封装，统一接口 | 可切换 Provider |
| **Database** | 数据持久化 | SQLAlchemy ORM |

---

## 四、数据库设计

### 4.1 ER 关系图

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   Session  │       │   Message   │       │    User     │
├─────────────┤       ├─────────────┤       ├─────────────┤
│ id          │──┐    │ id          │       │ id          │
│ user_id     │  │    │ session_id  │◄──────│ external_id │
│ target_role │  ├───►│ role        │       │ created_at  │
│ created_at  │  │    │ content     │       └─────────────┘
│ updated_at  │  │    │ intent      │
└─────────────┘  │    │ created_at  │
       │         │    └─────────────┘
       │         │
       ▼         ▼
┌───────────────────────────────────────┐
│           EmotionRecord               │
├───────────────────────────────────────┤
│ id                                    │
│ session_id (FK)                       │
│ overall_state                         │
│ current_mood                          │
│ emotions (JSON)                       │
│ confidence                            │
│ demand_type                           │
│ support_intensity                     │
│ created_at                            │
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│            SkillRecord                │
├───────────────────────────────────────┤
│ id                                    │
│ session_id (FK)                       │
│ skill_name                            │
│ level (0-100)                         │
│ required_level                        │
│ category                              │
│ source                                │
│ created_at                            │
└───────────────────────────────────────┘
```

### 4.2 表结构定义

```sql
-- sessions 表: 会话信息
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    target_role TEXT,
    current_phase VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- messages 表: 对话历史
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    intent VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- emotion_records 表: 情绪记录
CREATE TABLE emotion_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    overall_state VARCHAR(20),
    current_mood VARCHAR(50),
    emotions TEXT,
    confidence REAL,
    demand_type VARCHAR(50),
    support_intensity VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- skill_records 表: 技能记录
CREATE TABLE skill_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    skill_name VARCHAR(100) NOT NULL,
    level INTEGER,
    required_level INTEGER,
    category VARCHAR(50),
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

### 4.3 SQLAlchemy 模型

```python
# db/models.py
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    target_role = Column(String)
    current_phase = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    emotion_records = relationship("EmotionRecord", back_populates="session", cascade="all, delete-orphan")
    skill_records = relationship("SkillRecord", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    intent = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="messages")

class EmotionRecord(Base):
    __tablename__ = "emotion_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    overall_state = Column(String(20))
    current_mood = Column(String(50))
    emotions = Column(JSON)
    confidence = Column(Float)
    demand_type = Column(String(50))
    support_intensity = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="emotion_records")

class SkillRecord(Base):
    __tablename__ = "skill_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    skill_name = Column(String(100), nullable=False)
    level = Column(Integer)
    required_level = Column(Integer)
    category = Column(String(50))
    source = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="skill_records")
```

---

## 五、API 设计

### 5.1 API 端点总览

| 端点 | 方法 | 说明 |
|------|------|------|
| `POST /api/chat/{session_id}` | POST | SSE 流式对话 (核心) |
| `POST /api/sessions` | POST | 创建会话 |
| `GET /api/sessions/{session_id}` | GET | 获取会话信息 |
| `GET /api/emotion/trends/{session_id}` | GET | 获取情绪趋势 |
| `GET /api/skill-assessment/{session_id}` | GET | 获取技能评估 |
| `PATCH /api/skill-assessment/{session_id}` | PATCH | 更新技能评估 |
| `GET /api/resume/{session_id}` | GET | 获取简历信息 |
| `PATCH /api/resume/{session_id}` | PATCH | 更新简历信息 |

### 5.2 SSE 流式对话接口

**端点**: `POST /api/chat/{session_id}`

**请求体**:
```json
{
  "message": "我最近找工作很焦虑怎么办？"
}
```

**SSE 事件流**:

```text
event: start
data: {"type": "start", "session_id": "xxx"}

event: think_status
data: {"type": "think_status", "status": "正在分析意图...", "phase": "intent"}

event: think_status
data: {"type": "think_status", "status": "情绪检测完成", "phase": "emotion"}

event: think_complete
data: {"type": "think_complete", "metadata": {"intent": "emotional_support", "emotions": ["焦虑", "压力"]}}

event: sentence
data: {"type": "sentence", "index": 0, "content": "理解你的焦虑，", "emotion": "empathy", "is_last": false}

event: sentence
data: {"type": "sentence", "index": 1, "content": "这是很多求职者都会有的情绪。", "emotion": "neutral", "is_last": false}

event: sentence
data: {"type": "sentence", "index": 2, "content": "我们来分析下具体情况吧。", "emotion": "neutral", "is_last": true}

event: done
data: {"type": "done"}
```

### 5.3 请求/响应模型

```python
# api/schemas.py
from pydantic import BaseModel
from typing import Optional, List

class CreateSessionRequest(BaseModel):
    user_id: str
    target_role: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    target_role: Optional[str]
    created_at: str

class ChatRequest(BaseModel):
    message: str

class EmotionTrendsResponse(BaseModel):
    session_id: str
    current_mood: Optional[str]
    trend: str
    consecutive_negative: int
    negative_ratio: float
    needs_intervention: bool
    reason: str
    history: List[dict]

class SkillItem(BaseModel):
    name: str
    level: int
    required: int
    category: str

class GapItem(BaseModel):
    skill: str
    gap: int
    suggestion: str

class SkillAssessmentResponse(BaseModel):
    session_id: str
    target_role: str
    match_rate: int
    skills: List[SkillItem]
    gaps: List[GapItem]

class ResumeData(BaseModel):
    grade_level: Optional[str]
    major: Optional[str]
    school: Optional[str]
    target_role: Optional[str]
    job_search_stage: Optional[str]
    skill_focus: List[str]
    common_concerns: List[str]
    background_summary: Optional[str]

class ResumeResponse(BaseModel):
    session_id: str
    data: ResumeData
    last_updated: str
```

---

## 六、核心模块设计

### 6.1 LLM 适配器

支持多 LLM Provider 切换，当前配置 DeepSeek。

```python
# llm/base.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator

class LLMAdapter(ABC):
    @abstractmethod
    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def invoke(self, prompt: str) -> str:
        pass
```

```python
# llm/deepseek.py
from .base import LLMAdapter
import os
import httpx
from typing import AsyncGenerator
import json

class DeepSeekAdapter(LLMAdapter):
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.api_url = "https://api.deepseek.com/chat/completions"

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.api_url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if "choices" in chunk:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]

    async def invoke(self, prompt: str) -> str:
        chunks = []
        async for chunk in self.stream(prompt):
            chunks.append(chunk)
        return "".join(chunks)
```

```python
# llm/registry.py
from .base import LLMAdapter
from .deepseek import DeepSeekAdapter
import os

LLM_REGISTRY = {
    "deepseek": DeepSeekAdapter,
    "openai": None,
    "claude": None,
}

def get_llm_adapter(name: str = None) -> LLMAdapter:
    name = name or os.getenv("LLM_PROVIDER", "deepseek")
    adapter_class = LLM_REGISTRY.get(name)
    if adapter_class is None:
        raise ValueError(f"LLM adapter '{name}' not implemented yet")
    return adapter_class()
```

### 6.2 文本碎片化器

```python
# utils/text_fragmenter.py
import re
from dataclasses import dataclass
from typing import List
import random

@dataclass
class Sentence:
    content: str
    emotion: str
    delay_ms: int

class TextFragmenter:
    BOUNDARY_PATTERN = re.compile(r'[。？！\.]+')

    EMOTION_KEYWORDS = {
        'empathy': ['理解', '明白', '感同身受', '心疼'],
        'excited': ['太棒了', '太好了', '恭喜', '厉害', '加油'],
        'serious': ['注意', '建议', '必须', '一定', '重要'],
    }

    def split(self, text: str) -> List[Sentence]:
        sentences = []
        parts = self.BOUNDARY_PATTERN.split(text)
        boundaries = [m.end() for m in self.BOUNDARY_PATTERN.finditer(text)]

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            if i < len(boundaries):
                punctuation = ""
                for b in boundaries:
                    if b > (len("".join(parts[:i])) + len(part)):
                        punctuation = text[b-1]
                        break
                if punctuation:
                    part += punctuation

            emotion = self.detect_emotion(part)
            delay = self.calculate_delay(part, emotion)

            sentences.append(Sentence(content=part, emotion=emotion, delay_ms=delay))

        return sentences

    def detect_emotion(self, sentence: str) -> str:
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            if any(kw in sentence for kw in keywords):
                return emotion
        return 'neutral'

    def calculate_delay(self, sentence: str, emotion: str) -> int:
        length = len(sentence)
        base_delay = length * 50

        multipliers = {
            'empathy': 1.5,
            'excited': 0.7,
            'serious': 1.2,
            'neutral': 1.0,
        }

        delay = base_delay * multipliers.get(emotion, 1.0)
        jitter = 0.8 + random.random() * 0.4
        delay *= jitter

        return int(max(200, min(2000, delay)))
```

### 6.3 简单 Agent

Phase 1 实现，后续可扩展为 MultiAgent Orchestrator。

```python
# agent/simple_agent.py
from typing import AsyncGenerator, List, Optional
from dataclasses import dataclass

@dataclass
class ConversationContext:
    session_id: str
    user_id: str
    history: List[dict]
    target_role: Optional[str] = None

@dataclass
class AgentResult:
    content: str
    intent: str
    emotions: List[str]
    metadata: dict

class SimpleAgent:
    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    async def process(self, context: ConversationContext, user_message: str) -> AgentResult:
        intent = await self._analyze_intent(user_message)
        emotions = await self._detect_emotions(user_message)
        content = await self._generate_response(context, intent, emotions, user_message)

        return AgentResult(
            content=content,
            intent=intent,
            emotions=emotions,
            metadata={
                "target_role": context.target_role,
                "history_length": len(context.history)
            }
        )

    async def _analyze_intent(self, message: str) -> str:
        prompt = f"分析用户意图，分类如下：career_advice, skill_assessment, emotional_support, resume_help, job_search, casual_chat\n\n用户消息: {message}"
        return await self.llm.invoke(prompt)

    async def _detect_emotions(self, message: str) -> List[str]:
        prompt = f"分析用户情绪，返回情绪关键词列表：\n\n用户消息: {message}"
        result = await self.llm.invoke(prompt)
        return result.split(",") if result else []

    async def _generate_response(self, context, intent, emotions, user_message: str) -> str:
        prompt = f"""你是求职顾问AI，正在和用户对话。

用户消息: {user_message}
检测意图: {intent}
检测情绪: {emotions}

请生成回复，要求：
1. 句子不要太长，每句话表达一个完整意思
2. 使用"。"、"？"、"！"等标点自然断句
3. 语气要自然、温暖、像真人聊天

直接输出回复内容，不需要其他说明。"""
        chunks = []
        async for chunk in self.llm.stream(prompt):
            chunks.append(chunk)
        return "".join(chunks)
```

### 6.4 Agent 扩展接口

为后续多 Agent 架构预留的扩展接口。

```python
# agent/base.py
from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    @abstractmethod
    async def execute(self, input_data: Any) -> Any:
        pass

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
```

---

## 七、项目结构

```
backend/
├── main.py                      # FastAPI 入口
├── requirements.txt             # 依赖清单
├── .env.example                 # 环境变量示例
├── Dockerfile                   # 容器化配置 (后期)
│
├── api/                         # API 层
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py             # SSE 对话路由
│   │   ├── sessions.py         # 会话路由
│   │   ├── emotion.py          # 情绪路由
│   │   ├── skill.py            # 技能路由
│   │   └── resume.py           # 简历路由
│   ├── schemas.py              # Pydantic 模型
│   └── deps.py                 # 依赖注入
│
├── services/                   # 业务逻辑层
│   ├── __init__.py
│   ├── chat_service.py         # 对话服务
│   └── analysis_service.py     # 分析服务
│
├── agent/                      # Agent 层
│   ├── __init__.py
│   ├── base.py                 # Agent 基类 (扩展接口)
│   ├── simple_agent.py         # Phase 1: 简单 Agent
│   ├── intent_analyzer.py      # 意图分析
│   ├── emotion_detector.py     # 情绪检测
│   └── response_generator.py   # 回复生成
│   │
│   │   # 后续扩展
│   ├── orchestrator.py         # MultiAgent 编排器
│   ├── router_agent.py         # 路由 Agent
│   ├── dialogue_agent.py       # 对话 Agent
│   └── analyzer_agents/        # 分析类 Agent
│
├── llm/                        # LLM 适配层
│   ├── __init__.py
│   ├── base.py                 # 适配器基类
│   ├── deepseek.py            # DeepSeek 实现
│   ├── openai.py              # OpenAI 预留
│   ├── claude.py              # Claude 预留
│   └── registry.py            # 适配器注册
│
├── utils/                      # 工具层
│   ├── __init__.py
│   ├── text_fragmenter.py     # 文本碎片化
│   ├── emotion_rhythm.py      # 情感节奏
│   └── config.py               # 配置管理
│
└── db/                         # 数据库层
    ├── __init__.py
    ├── database.py             # 数据库连接
    ├── models.py               # ORM 模型
    └── crud.py                 # CRUD 操作
```

---

## 八、环境配置

```bash
# .env.example
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-chat
LLM_PROVIDER=deepseek
DATABASE_URL=sqlite:///./data/app.db
HOST=0.0.0.0
PORT=8080
DEBUG=true
```

---

## 九、开发计划

### Phase 1: MVP 实现 (预计 3-5 天)

| 序号 | 任务 | 描述 | 优先级 |
|------|------|------|--------|
| 1 | 项目初始化 | 创建 FastAPI 项目结构，安装依赖 | P0 |
| 2 | 数据库配置 | SQLite + SQLAlchemy ORM | P0 |
| 3 | LLM 适配器 | DeepSeek API 集成，支持流式 | P0 |
| 4 | 会话管理 | 创建/获取会话 API | P0 |
| 5 | SSE 对话接口 | POST /api/chat/{id} 流式响应 | P0 |
| 6 | 文本碎片化 | Sentence 拆分 + 情感检测 | P0 |
| 7 | 简单 Agent | Intent + Emotion + Response | P1 |
| 8 | 情绪 API | GET /api/emotion/trends/{id} | P1 |
| 9 | 技能 API | GET/PATCH /api/skill-assessment/{id} | P1 |
| 10 | 简历 API | GET/PATCH /api/resume/{id} | P2 |
| 11 | 前端对接 | 适配现有前端 WebSocket → SSE | P1 |

**里程碑**: 后端可独立运行，前端完成对话功能

---

### Phase 2: 功能完善 (预计 3-5 天)

| 序号 | 任务 | 描述 | 优先级 |
|------|------|------|--------|
| 1 | 提示词优化 | 精细化对话提示词 | P1 |
| 2 | 技能提取 | 从对话中自动提取技能 | P1 |
| 3 | 情绪趋势 | 完善情绪分析和历史追踪 | P1 |
| 4 | 错误处理 | 完善异常处理和重试机制 | P1 |
| 5 | 日志系统 | 结构化日志记录 | P2 |
| 6 | 单元测试 | 核心模块测试覆盖 | P2 |

**里程碑**: 功能完整，可小范围测试

---

### Phase 3: 架构扩展 (后续迭代)

| 序号 | 任务 | 描述 | 状态 |
|------|------|------|------|
| 1 | MultiAgent 编排器 | 完整的多 Agent 协作框架 | 待开发 |
| 2 | 知识库集成 | RAG 向量检索支持 | 待开发 |
| 3 | LLM 切换 | OpenAI/Claude 适配 | 预留接口 |
| 4 | 数据库迁移 | SQLite → PostgreSQL | 预留 |
| 5 | Docker 部署 | 容器化 + CI/CD | 预留 |
| 6 | 缓存层 | Redis 会话缓存 | 待规划 |

---

## 十、待讨论事项

以下事项需要在开发过程中进一步确认：

1. **DeepSeek API 额度**: 是否已有 API Key？
2. **提示词设计**: 是否需要提前设计，还是边开发边优化？
3. **技能库**: 是否有预设的技能列表用于匹配？
4. **情绪分析**: 使用 LLM 做情绪判断还是独立的情绪模型？
5. **数据存储**: 是否需要用户注册登录系统？

---

## 附录 A: 前端 API 对接说明

前端现有 `useWebSocket.ts` 需要调整为 SSE 方式：

```typescript
// useSSEChat.ts (替代 useWebSocket.ts)
export function useSSEChat(sessionId: string) {
  const sendMessage = async (content: string) => {
    const response = await fetch(`/api/chat/${sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: content }),
    })

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()

    while (reader) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value)
      // 解析 SSE 事件...
    }
  }

  return { sendMessage }
}
```

---

## 附录 B: 参考资料

- FastAPI SSE: https://fastapi.tiangolo.com/advanced/using-alternative-fastapi/
- SSE 规范: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- DeepSeek API: https://platform.deepseek.com/

---

**文档状态**: 初稿，待评审
**下一步**: 确定开发优先级，启动 Phase 1
