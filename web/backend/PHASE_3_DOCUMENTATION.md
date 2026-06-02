# Phase 3 开发文档：CareerAgent + LangChain + MCP 完整集成

## 概述

Phase 3 实现了完整的 AI 架构集成，将 LangChain 框架、CareerAgent 智能体和 MCP 协议与现有 FastAPI 后端无缝对接。

## 架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         前端应用                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ SSE 流式响应
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI 后端服务                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  API 路由层                                   │ │
│  │  - /careers/chat/{session_id} (SSE 流式)                     │ │
│  │  - /careers/config                                           │ │
│  │  - /careers/memory/{session_id}                              │ │
│  └────────────────────────┬────────────────────────────────────┘ │
│                           │                                      │
│  ┌────────────────────────▼────────────────────────────────────┐ │
│  │                  Agent 工厂 (agent_factory.py)               │ │
│  │  - 统一接口创建 SimpleAgent / CareerAgent                    │ │
│  │  - 优雅降级：依赖缺失时回退到 SimpleAgent                     │ │
│  └────────────────────────┬────────────────────────────────────┘ │
│                           │                                      │
│          ┌────────────────┴────────────────┐                     │
│          │                                 │                     │
│  ┌───────▼────────┐            ┌──────────▼──────────┐           │
│  │  SimpleAgent   │            │    CareerAgent      │           │
│  │  (Phase 1/2)   │            │  (完整 LangChain)   │           │
│  └────────────────┘            └──────────┬──────────┘           │
│                                            │                      │
│                              ┌─────────────▼──────────────┐      │
│                              │    LangChain 框架           │      │
│                              │  - ChatOpenAI (DeepSeek)   │      │
│                              │  - InMemoryChatHistory     │      │
│                              │  - Tool Calling            │      │
│                              └─────────────┬──────────────┘      │
│                                            │                      │
│                              ┌─────────────▼──────────────┐      │
│                              │     MCP 客户端             │      │
│                              │  langchain-mcp-adapters    │      │
│                              └─────────────┬──────────────┘      │
│                                            │ HTTP                 │
│                              ┌─────────────▼──────────────┐      │
│                              │     MCP 服务器             │      │
│                              │  (FastMCP, 端口 8001)      │      │
│                              │  - resume_skill 工具       │      │
│                              └────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
backend/
├── careers/                          # Phase 3 新增模块
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── career_agent.py          # CareerAgent 核心实现
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py                # MCP 服务器启动
│   │   ├── prompts/
│   │   │   └── __init__.py
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── resume_tool.py       # 简历工具实现
│   │       └── generate_resume_pdf.py
│   └── prompts/
│       ├── __init__.py
│       └── career/
│           ├── __init__.py
│           ├── system_prompt.yaml   # 系统提示词
│           └── extraction.yaml      # 信息提取提示词
├── api/routes/
│   └── careers.py                   # Phase 3 更新的 API 路由
├── agent_factory.py                 # Phase 3 更新的 Agent 工厂
├── requirements.txt                 # Phase 3 更新的依赖
├── PHASE_3_DOCUMENTATION.md         # 本文档
└── ...
```

## 核心组件详解

### 1. CareerAgent (careers/agents/career_agent.py)

**功能特性：**
- 基于 LangChain 的完整 Agent 实现
- 支持 DeepSeek API 调用
- 集成 MCP 工具调用能力
- 双重记忆系统：
  - 短期记忆：InMemoryChatHistory
  - 长期记忆：JSON 文件持久化
- 信息自动提取：使用 LLM 从对话中提取用户信息、偏好、目标等
- 流式对话输出

**关键方法：**
```python
class CareerAgent:
    def __init__(api_key, mcp_url, memory_file, use_mcp)
    async def chat(user_input) -> str
    async def stream_chat(user_input) -> AsyncIterator[str]
    async def connect_mcp() -> List[BaseTool]
    def get_memory_summary() -> Dict
    def clear_memory() -> None
```

**依赖检查：**
代码包含完整的可用性检查，当 LangChain 依赖缺失时会优雅降级。

### 2. Agent 工厂 (agent_factory.py)

**设计模式：** 工厂模式 + 单例模式

**职责：**
- 统一创建不同类型的 Agent
- 管理依赖注入
- 优雅降级机制

**使用方式：**
```python
from agent_factory import get_agent_factory

factory = get_agent_factory()
agent = factory.create_agent("career", use_mcp=True)
```

### 3. API 路由 (api/routes/careers.py)

**端点列表：**

| 端点 | 方法 | 描述 |
|------|------|------|
| `/careers/chat/{session_id}` | POST | SSE 流式对话 |
| `/careers/config` | GET | 获取配置信息 |
| `/careers/memory/{session_id}` | GET | 获取记忆摘要 |
| `/careers/memory/{session_id}` | DELETE | 清除记忆 |

### 4. MCP 服务器 (careers/mcp/server.py)

**功能：**
- 使用 FastMCP 框架实现 MCP 协议
- 暴露简历生成、优化等工具
- 使用 Streamable HTTP 传输（端口 8001）

**启动方式：**
```bash
cd backend
python -m careers.mcp.server
```

## 依赖管理

### requirements.txt 更新内容

```txt
# LangChain 相关依赖
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-core>=0.2.0
langchain-mcp-adapters>=0.1.0

# MCP 相关依赖
fastmcp>=0.7.0
mcp>=1.1.0

# 其他工具依赖
pyyaml>=6.0
reportlab>=4.0.0
```

### 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

## 配置说明

### 环境变量 (.env)

```env
# DeepSeek API 配置
DEEPSEEK_API_KEY=your_api_key_here

# MCP 配置
MCP_URL=http://localhost:8001/mcp
USE_MCP=true
```

## 使用指南

### 1. 启动 MCP 服务器（可选）

```bash
# 终端 1
cd backend
python -m careers.mcp.server
```

### 2. 启动 FastAPI 后端

```bash
# 终端 2
cd backend
uvicorn main:app --reload
```

### 3. 测试 API

访问 http://localhost:8000/docs 查看 Swagger 文档。

**示例：发送聊天请求**

```bash
curl -X POST "http://localhost:8000/careers/chat/test-session" \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，我正在找工作"}'
```

## 记忆系统

### 长期记忆存储格式

```json
{
  "user_info": {
    "2024-01-15 10:30:00": "计算机科学专业"
  },
  "preferences": {},
  "emotions": {},
  "goals": [],
  "important_events": [],
  "conversation_summary": [],
  "last_interaction": "2024-01-15 10:35:00"
}
```

### 记忆 API 使用

```python
# 获取记忆摘要
GET /careers/memory/{session_id}

# 清除记忆
DELETE /careers/memory/{session_id}
```

## 降级策略

系统设计了完整的优雅降级机制：

1. **LangChain 依赖缺失** → 回退到 SimpleAgent
2. **MCP 连接失败** → CareerAgent 使用内置知识，不调用工具
3. **DeepSeek API 失败** → 捕获异常并返回友好提示

## 测试建议

### 1. 基础功能测试

- [ ] 聊天接口正常响应
- [ ] 流式输出工作正常
- [ ] 记忆读写正常
- [ ] 配置接口返回正确信息

### 2. 降级测试

- [ ] 不安装 LangChain 依赖 → 验证回退到 SimpleAgent
- [ ] 不启动 MCP 服务器 → 验证 CareerAgent 仍能工作
- [ ] 无效的 API Key → 验证错误处理

### 3. 集成测试

- [ ] 完整流程：从用户输入到 AI 响应
- [ ] 记忆持久化：重启服务后记忆仍存在
- [ ] 多会话：不同 session_id 有独立记忆

## 注意事项

1. **保持 LangChain 架构**：本实现完全保留了原始 LangChain 架构，未做修改
2. **MCP 工具**：当前仅实现 resume_skill，可扩展添加更多工具
3. **性能优化**：生产环境建议使用 Redis 替代 JSON 文件存储记忆
4. **安全**：生产环境不要将 .env 文件提交到版本控制

## 后续扩展方向

- [ ] 添加更多 MCP 工具（职业分析、面试模拟等）
- [ ] 实现记忆的向量检索
- [ ] 添加用户认证
- [ ] 支持多语言
- [ ] 添加使用统计和分析
