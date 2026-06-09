# Phase 1 - Career Agent 适配层完成

## 概述

Phase 1 已成功完成！已成功将 OpenCareer 的 AI 架构（CareerAgent + LangChain + MCP）集成到 Web 后端中，**保持 LangChain 架构原样**，通过适配器层实现兼容。

## 目录结构

```
web/backend/
├── careers/                          # 新增：Career Agent 模块
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── career_agent.py          # LangChain CareerAgent (原样)
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py                 # MCP Server
│   │   ├── prompts/
│   │   │   └── __init__.py
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── resume_tool.py         # 简历工具
│   │       └── generate_resume_pdf.py # PDF 生成器
│   └── prompts/
│       ├── __init__.py
│       └── career/
│           ├── __init__.py
│           ├── system_prompt.yaml     # 系统提示词
│           └── extraction.yaml        # 信息提取提示词
├── agent_factory.py                  # Agent 工厂（Phase 2 增强）
├── agent_adapter.py                  # Agent 适配器（Phase 2 增强）
├── careers_config.py                 # 配置文件
├── requirements.txt                  # 依赖已更新
└── CAREERS_INTEGRATION.md            # 集成说明（原始）
```

## 核心组件详解

### 1. CareerAgent ([careers/agents/career_agent.py](file:///F:/AI%20Career%20Companion/OpenCareer/demo/web/backend/careers/agents/career_agent.py))

**核心功能：**
- LangChain Agent 实现，支持 MCP 工具调用
- 长期记忆 + 短期记忆管理
- 流式输出支持
- LLM 驱动的信息提取

**关键方法：**
```python
CareerAgent(
    api_key: Optional[str] = None,
    mcp_url: str = "http://localhost:8001/mcp",
    memory_file: str = "career_memory.json",
    use_mcp: bool = True
)

# 主交互方法
await agent.chat(user_input: str) -> str
async agent.stream_chat(user_input: str) -> AsyncIterator[str]
agent.get_memory_summary() -> Dict[str, int]
agent.clear_memory() -> None
```

**系统提示词：** [system_prompt.yaml](file:///F:/AI%20Career%20Companion/OpenCareer/demo/web/backend/careers/prompts/career/system_prompt.yaml)

### 2. MCP Server ([careers/mcp/server.py](file:///F:/AI%20Career% Companion/OpenCareer/demo/web/backend/careers/mcp/server.py))

**功能：**
- 使用 FastMCP 暴露职业工具
- Streamable HTTP 传输协议
- 自动注册 MCP 工具

**工具列表：**
- `resume_skill`: 简历生成、优化、ATS 检查、PDF 导出

**启动方式：**
```bash
cd web/backend
python -m careers.mcp.server
```

### 3. Agent 工厂 ([agent_factory.py](file:///F:/AI%20Career%20Companion/OpenCareer/demo/web/backend/agent_factory.py))

**功能：**
- 根据配置创建不同类型的 Agent
- 单例模式，全局缓存
- 支持回退机制

**Agent 类型：**
```python
class AgentType(str, Enum):
    SIMPLE = "simple"      # 现有 SimpleAgent
    CAREER = "career"      # 新增 CareerAgent
```

**使用示例：**
```python
from agent_factory import get_agent_factory, AgentType

factory = get_agent_factory()
agent = factory.create_agent(AgentType.CAREER)
```

### 4. Agent 适配器 ([agent_adapter.py](file:///F:/AI%20Career%20Companion/OpenCareer/demo/web/backend/agent_adapter.py))

**功能：**
- 统一不同 Agent 的接口
- SSE 事件格式适配
- 情感检测

**主要方法：**
```python
adapter = AgentAdapter(agent)
async for event in adapter.chat(session_id, messages):
    # 处理 SSE 事件
    pass
```

## 配置管理 ([careers_config.py](file:///F:/AI%20Career%20Companion/OpenCareer/demo/web/backend/careers_config.py))

### 环境变量

在 `.env` 文件中添加：

```env
# Career Agent 配置
CAREER_AGENT_TYPE=simple              # "simple" 或 "career"
CAREER_USE_MCP=false                  # 是否使用 MCP (需要 MCP Server)
CAREER_MCP_URL=http://localhost:8001/mcp
CAREER_MCP_HOST=127.0.0.1
CAREER_MCP_PORT=8001

# 记忆配置
CAREER_MEMORY_DIR=./data/memory
CAREER_MEMORY_FILE=career_memory.json

# DeepSeek API (CareerAgent 需要)
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-chat
```

### 验证配置

```python
from careers_config import config

is_valid, errors = config.validate_config()
print(f"配置有效: {is_valid}")
print(f"错误: {errors}")
```

## 依赖安装

requirements.txt 已更新，添加以下依赖：

```
# LangChain & MCP (for CareerAgent)
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-mcp-adapters>=0.1.0
fastmcp>=0.7.0
pyyaml>=6.0

# PDF Generation
reportlab>=4.0.0
```

**安装方式：**
```bash
cd web/backend
pip install -r requirements.txt
```

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Routes: chat, careers, sessions, emotion, skill, ...   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│     SimpleAgent         │          │     AgentFactory        │
│   (现有后端 Agent)      │          │      (新架构)           │
└─────────────────────────┘          └─────────────────────────┘
                                                    │
                                    ┌───────────────┴───────────────┐
                                    │                               │
                                    ▼                               ▼
                            ┌──────────────┐             ┌─────────────────────┐
                            │ SimpleAgent │             │    CareerAgent       │
                            │   (当前)    │             │ (LangChain + MCP)    │
                            └─────────────┘             └─────────────────────┘
                                                                   │
                                                    ┌──────────────┴──────────────┐
                                                    │                              │
                                                    ▼                              ▼
                                        ┌─────────────────┐            ┌─────────────────┐
                                        │  MCP Server     │            │  DeepSeek LLM   │
                                        │ (FastMCP)       │            │                 │
                                        └─────────────────┘            └─────────────────┘
                                                    │
                                        ┌───────────┴───────────┐
                                        │                       │
                                        ▼                       ▼
                                ┌──────────────┐     ┌──────────────────┐
                                │ resume_tool  │     │ (其他工具...)     │
                                └──────────────┘     └──────────────────┘
```

## 使用方式

### 方式 1: 使用 SimpleAgent（默认，保持不变）

```python
# 现有代码无需修改
from agent.simple_agent import SimpleAgent
from llm.registry import get_llm_adapter

llm = get_llm_adapter()
agent = SimpleAgent(llm)

async for event in agent.chat(session_id, history):
    print(event)
```

### 方式 2: 启用 CareerAgent

```python
from agent_factory import get_agent_factory, AgentType
from agent_adapter import create_adapter

# 创建 CareerAgent
factory = get_agent_factory()
agent = factory.create_agent(AgentType.CAREER)

# 使用适配器统一接口
adapter = create_adapter(agent)

# 使用兼容接口
async for event in adapter.chat(session_id, messages):
    print(event)
```

### 方式 3: 运行 MCP Server（可选）

```bash
# 终端 1: 启动 MCP Server
cd web/backend
python -m careers.mcp.server

# 终端 2: 启动 FastAPI
python main.py
```

## 关键特性

✅ **保持 LangChain 架构原样** - 所有 LangChain、MCP 相关代码未做修改

✅ **向后兼容** - 默认仍使用 SimpleAgent，不影响现有功能

✅ **渐进式升级** - 可以逐步切换到 CareerAgent

✅ **工具支持** - 简历生成、优化、ATS 检查、PDF 导出

✅ **记忆系统** - 长期记忆 + 短期记忆，支持信息提取

## Phase 1 完成状态

| 组件 | 状态 | 说明 |
|------|------|------|
| CareerAgent | ✅ 完成 | LangChain Agent，保持原样 |
| MCP Server | ✅ 完成 | FastMCP 实现 |
| 简历工具 | ✅ 完成 | 完整功能实现 |
| Agent 工厂 | ✅ 完成 | 支持多类型 Agent |
| Agent 适配器 | ✅ 完成 | 统一接口 |
| 配置文件 | ✅ 完成 | 环境变量支持 |
| 依赖更新 | ✅ 完成 | requirements.txt 已更新 |
| 文档 | ✅ 完成 | 本文档 |

## 注意事项

1. **保持 LangChain 架构原样** - 所有 LangChain 相关代码未做修改
2. **向后兼容** - 默认仍使用 SimpleAgent，不影响现有功能
3. **渐进式升级** - 可以逐步切换到 CareerAgent
4. **依赖安装** - Phase 3 集成前需要安装新增依赖

## 下一步

- **Phase 2**: API 路由扩展（已完成）✅
- **Phase 3**: 完整 CareerAgent 集成（需要安装 LangChain/MCP 依赖）
- **Phase 4**: 记忆管理 UI 集成

## Phase 1 已圆满完成！🎉

**保持 LangChain 架构原样，通过适配器实现兼容**
