# Phase 4 开发文档：MultiAgent 编排 + 知识库 + LLM 切换 + Docker 部署

## 概述

Phase 4 实现了以下核心功能：
1. **MultiAgent 编排器** - 自动根据用户意图选择最合适的 Agent
2. **知识库集成** - 使用 FAISS 向量数据库实现文档检索
3. **LLM 切换支持** - 支持 DeepSeek、OpenAI、Claude 三种 LLM 提供者
4. **Docker 部署** - 提供完整的容器化部署方案

## 架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         前端应用                                        │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ SSE 流式响应
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FastAPI 后端服务                                     │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                     API 路由层                                     │ │
│  │  - /orchestrator/chat/{session_id}  (智能路由)                    │ │
│  │  - /orchestrator/delegate/{session_id} (直接委托)                 │ │
│  │  - /llm/providers                    (LLM 提供者列表)              │ │
│  │  - /llm/health/{provider}           (健康检查)                   │ │
│  │  - /knowledge/search                 (知识库搜索)                 │ │
│  │  - /knowledge/query                  (知识库查询)                 │ │
│  │  - /knowledge/add                    (添加文档)                   │ │
│  └────────────────────────┬─────────────────────────────────────────┘ │
│                           │                                          │
│  ┌────────────────────────▼─────────────────────────────────────────┐ │
│  │                  Agent Orchestrator                              │ │
│  │  - 意图分类 (Intent Classification)                              │ │
│  │  - Agent 选择路由                                               │ │
│  │  - 多 Agent 协调                                                │ │
│  └────────────────────────┬─────────────────────────────────────────┘ │
│                           │                                          │
│          ┌────────────────┼────────────────┐                         │
│          ▼                ▼                ▼                         │
│  ┌───────────┐    ┌───────────┐    ┌─────────────┐                  │
│  │ General   │    │  Career   │    │ Knowledge   │                  │
│  │  Agent    │    │   Agent   │    │    Base     │                  │
│  └─────┬─────┘    └─────┬─────┘    └──────┬──────┘                  │
│        │                │                  │                         │
│        ▼                ▼                  ▼                         │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │                    LLM 适配器层                             │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                   │     │
│  │  │ DeepSeek │ │  OpenAI  │ │  Claude  │                   │     │
│  │  └──────────┘ └──────────┘ └──────────┘                   │     │
│  └─────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
backend/
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── simple_agent.py
│   └── orchestrator.py           # Phase 4 新增：MultiAgent 编排器
├── llm/
│   ├── __init__.py
│   ├── base.py
│   ├── deepseek.py
│   ├── openai.py                 # Phase 4 新增：OpenAI 适配器
│   ├── claude.py                 # Phase 4 新增：Claude 适配器
│   └── registry.py               # Phase 4 更新：支持多 LLM 切换
├── knowledge/                    # Phase 4 新增：知识库模块
│   ├── __init__.py
│   └── knowledge_base.py         # FAISS 向量数据库实现
├── api/routes/
│   ├── __init__.py
│   └── orchestrator.py           # Phase 4 新增：编排器 API 路由
├── requirements.txt              # Phase 4 更新：新增依赖
├── Dockerfile                    # Phase 4 新增：后端 Dockerfile
└── PHASE_4_DOCUMENTATION.md      # 本文档

web/
├── docker-compose.yml            # Phase 4 新增：Docker Compose 配置
└── front/
    └── Dockerfile                # Phase 4 新增：前端 Dockerfile
```

## 核心组件详解

### 1. MultiAgent 编排器 (agents/orchestrator.py)

**功能特性：**
- 意图分类：根据用户输入自动识别意图
- Agent 路由：根据意图选择最合适的 Agent
- 优雅降级：当目标 Agent 不可用时回退到通用 Agent
- 支持直接委托：可强制指定使用某个 Agent

**意图分类规则：**

| 意图类型 | 触发关键词 | 目标 Agent |
|----------|------------|-----------|
| GENERAL | 默认 | SimpleAgent |
| CAREER_ADVISOR | 找工作、求职、面试、薪资、职业发展、跳槽 | CareerAgent |
| RESUME_EXPERT | 简历、CV、优化简历、ATS | CareerAgent |
| EMOTION_SUPPORT | 焦虑、压力、迷茫、沮丧、烦恼 | CareerAgent |
| SKILL_ASSESSOR | 技能、学习、掌握、技术、能力 | CareerAgent |

**使用方式：**
```python
from agents.orchestrator import get_orchestrator

orchestrator = get_orchestrator()

# 智能路由（自动选择 Agent）
async for token in orchestrator.route("我想优化简历"):
    print(token)

# 直接委托（指定 Agent）
async for token in orchestrator.delegate("你好", "general"):
    print(token)
```

### 2. LLM 适配器扩展

**支持的 LLM 提供者：**

| 提供者 | 环境变量 | 依赖库 |
|--------|----------|--------|
| DeepSeek | DEEPSEEK_API_KEY | httpx (内置) |
| OpenAI | OPENAI_API_KEY | openai>=1.0.0 |
| Claude | ANTHROPIC_API_KEY | anthropic>=0.20.0 |

**配置方式：**

在 `.env` 文件中配置：
```env
# 默认 LLM 提供者
LLM_PROVIDER=deepseek

# DeepSeek 配置
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_MODEL=deepseek-chat

# OpenAI 配置（可选）
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_BASE_URL=https://api.openai.com/v1

# Claude 配置（可选）
ANTHROPIC_API_KEY=your_anthropic_key
CLAUDE_MODEL=claude-3-sonnet-20240229
```

**动态切换：**
```python
from llm.registry import get_llm_adapter, list_available_llms

# 获取所有可用的 LLM
providers = list_available_llms()

# 使用指定的 LLM
adapter = get_llm_adapter("openai")
response = await adapter.invoke([{"role": "user", "content": "Hello"}])
```

### 3. 知识库系统 (knowledge/knowledge_base.py)

**功能特性：**
- 使用 FAISS 向量数据库进行文档检索
- 支持文本嵌入生成（使用 OpenAI Embeddings）
- 文档添加、搜索、查询功能
- 索引持久化到磁盘

**使用方式：**
```python
from knowledge.knowledge_base import get_knowledge_base

kb = get_knowledge_base()

# 添加文档
kb.add_document("简历优化技巧：突出量化成果", {"source": "career_guide.txt"})

# 搜索知识库
results = kb.search("简历优化", k=5)

# 查询知识库（格式化输出）
answer = kb.query("如何优化简历？")
```

**索引存储位置：**
- `knowledge/faiss_index/index.faiss` - FAISS 索引文件
- `knowledge/faiss_index/documents.json` - 文档元数据

### 4. Docker 部署

**Docker Compose 服务：**

| 服务 | 端口 | 说明 |
|------|------|------|
| backend | 8000 | FastAPI 后端服务 |
| mcp-server | 8001 | MCP 工具服务器 |
| frontend | 5173 | 前端应用 |

**启动方式：**

```bash
# 在 web 目录下创建 .env 文件
cd web
cp backend/.env.example .env

# 编辑 .env 文件，添加 API Key

# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

**环境变量配置：**

在 `docker-compose.yml` 所在目录创建 `.env` 文件：
```env
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LLM_PROVIDER=deepseek
```

## API 端点清单

### Orchestrator API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/orchestrator/chat/{session_id}` | POST | 智能路由对话（自动选择 Agent） |
| `/api/orchestrator/delegate/{session_id}` | POST | 直接委托给指定 Agent |
| `/api/orchestrator/agents` | GET | 获取可用 Agent 列表 |

### LLM API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/llm/providers` | GET | 获取可用 LLM 提供者列表 |
| `/api/llm/health/{provider}` | GET | 检查指定 LLM 健康状态 |

### Knowledge API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/knowledge/search` | POST | 搜索知识库 |
| `/api/knowledge/query` | POST | 查询知识库（格式化） |
| `/api/knowledge/add` | POST | 添加文档到知识库 |
| `/api/knowledge/stats` | GET | 获取知识库统计信息 |
| `/api/knowledge/clear` | DELETE | 清空知识库 |

## 依赖更新

### requirements.txt 新增内容

```txt
# Phase 4 新增依赖

# OpenAI API 支持
openai>=1.0.0

# Claude API 支持
anthropic>=0.20.0

# FAISS 向量数据库（用于知识库）
faiss-cpu>=1.8.0

# 文档解析
python-docx>=0.8.11
pdfplumber>=0.10.0
```

### 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

## 使用指南

### 1. 本地开发

```bash
# 启动后端
cd backend
uvicorn main:app --reload

# 启动 MCP 服务器（可选）
python -m careers.mcp.server
```

### 2. Docker 部署

```bash
# 构建并启动
cd web
docker-compose up -d

# 停止服务
docker-compose down

# 查看日志
docker-compose logs -f
```

### 3. 测试 LLM 切换

```bash
# 测试 DeepSeek
curl -X GET "http://localhost:8000/api/llm/health/deepseek"

# 测试 OpenAI（需要配置 OPENAI_API_KEY）
curl -X GET "http://localhost:8000/api/llm/health/openai"

# 测试 Claude（需要配置 ANTHROPIC_API_KEY）
curl -X GET "http://localhost:8000/api/llm/health/claude"
```

### 4. 使用知识库

```bash
# 添加文档
curl -X POST "http://localhost:8000/api/knowledge/add" \
  -H "Content-Type: application/json" \
  -d '{"content": "面试技巧：提前准备常见问题", "metadata": {"source": "interview_guide.txt"}}'

# 搜索知识库
curl -X POST "http://localhost:8000/api/knowledge/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "面试准备", "max_results": 3}'

# 查询知识库
curl -X POST "http://localhost:8000/api/knowledge/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "如何准备面试？"}'
```

## 测试建议

### 1. MultiAgent 编排器测试

- [ ] 测试意图分类准确性
- [ ] 测试 Agent 路由功能
- [ ] 测试降级机制（Agent 不可用时）

### 2. LLM 切换测试

- [ ] 测试所有 LLM 提供者的健康检查
- [ ] 测试动态切换不同 LLM
- [ ] 测试 API Key 缺失时的错误处理

### 3. 知识库测试

- [ ] 测试文档添加功能
- [ ] 测试搜索准确性
- [ ] 测试索引持久化（重启服务后数据是否保留）

### 4. Docker 部署测试

- [ ] 测试 Docker Compose 启动
- [ ] 测试服务间通信
- [ ] 测试数据卷持久化

## 注意事项

1. **OpenAI Embeddings**：知识库功能需要 OpenAI API Key 用于生成文本嵌入
2. **FAISS 索引**：首次使用会自动创建索引文件，后续启动会加载已有索引
3. **Docker 网络**：确保 Docker 容器之间可以通信（使用 docker-compose 自动创建网络）
4. **环境变量安全**：不要将 API Key 提交到版本控制，使用 `.env` 文件管理

## 后续扩展方向

- [ ] 添加更多 LLM 提供者（如 Llama、Qwen 等开源模型）
- [ ] 实现知识库的增量更新和向量索引优化
- [ ] 添加 RAG（Retrieval-Augmented Generation）功能
- [ ] 实现 Agent 之间的信息共享和协作
- [ ] 添加分布式部署支持
