# LangChain + MCP 集成指南

本指南介绍如何在 OpenCareer 项目中使用 LangChain 与 MCP（Model Context Protocol）来构建智能代理系统。

## 架构概览

```
用户输入 → LangChain Agent → 智能工具选择 → MCP Skills → 执行结果 → Agent 整合 → 返回给用户
```

### 核心组件

1. **OpenCareerAgent** (`opencareer/langchain_mcp/agent.py`)
   - 主代理类，负责协调对话和工具调用
   - 集成了 LLM、记忆系统和工具加载器

2. **MCPSkillToolLoader** (`opencareer/langchain_mcp/tool_loader.py`)
   - 负责将 OpenCareer Skills 转换为 LangChain Tools
   - 提供技能的发现、加载和管理功能

3. **OpenCareerMemory** (`opencareer/langchain_mcp/memory.py`)
   - 统一的记忆系统
   - 整合对话历史、用户档案和长期记忆

4. **SkillTool** (`opencareer/mcp/tools.py`)
   - 将 BaseSkill 包装为 LangChain BaseTool 的适配器

## 快速开始

### 1. 基本使用

```python
import asyncio
from opencareer.langchain_mcp import create_opencareer_agent

async def main():
    # 创建代理
    agent = create_opencareer_agent()
    
    # 初始化（加载技能）
    await agent.initialize()
    
    # 对话
    response = await agent.chat("你好，我想优化我的简历")
    print(response['response'])
    
    # 保存状态
    agent.save_state()

asyncio.run(main())
```

### 2. 运行示例

```bash
cd demo
python langchain_mcp_example.py
```

### 3. 直接调用技能

```python
# 直接执行特定技能
result = await agent.execute_skill_direct(
    "resume_cn_career",
    {
        "action": "generate",
        "name": "张三",
        "target_role": "软件工程师",
        "experiences": [...]
    }
)
```

## 架构详解

### Skill → Tool 转换流程

```
BaseSkill (现有系统)
    ↓
SkillMetadata (描述、输入/输出 schema)
    ↓
create_langchain_tool_from_skill() (转换函数)
    ↓
SkillTool (LangChain BaseTool)
    ↓
AgentExecutor (使用)
```

### 记忆系统

记忆系统包含三个层次：

1. **对话历史** - 最近的 N 轮对话
2. **用户档案** - 用户的个人信息、目标等
3. **长期记忆** - 重要的事实和偏好

```python
# 更新用户档案
agent.update_user_profile(
    name="张三",
    target_role="软件工程师",
    experience_years="3年"
)

# 获取档案
profile = agent.get_user_profile()
```

## 自定义扩展

### 添加新的 Skill 到 LangChain

1. 确保你的 Skill 继承自 `BaseSkill`
2. 提供完整的 `SkillMetadata`，特别是 `input_schema`
3. 放置在 `opencareer/skills/` 目录下
4. 系统会自动发现并加载

### 自定义 Agent Prompt

```python
from opencareer.langchain_mcp import create_opencareer_agent

custom_prompt = """你是一个专业的职业顾问..."""

agent = create_opencareer_agent(
    system_prompt=custom_prompt
)
```

## 文件结构

```
demo/opencareer/langchain_mcp/
├── __init__.py          # 模块入口
├── agent.py             # OpenCareerAgent 主类
├── tool_loader.py       # Skill 到 Tool 的加载器
└── memory.py            # 记忆系统

demo/
├── langchain_mcp_example.py  # 使用示例
└── docs/
    └── LANGCHAIN_MCP_GUIDE.md  # 本文档
```

## 工作流程示例

### 场景：用户需要简历优化

1. **用户输入**: "我想优化我的简历，应聘软件工程师"
2. **Agent 理解**: 识别需要使用 resume_cn_career skill
3. **信息收集**: 可能询问用户更多细节（如果需要）
4. **工具调用**: 调用 resume_cn_career 的 optimize action
5. **结果整合**: 将技能返回的结果转换为自然语言
6. **返回**: "我已经帮你优化了简历，主要改进了..."

## 最佳实践

1. **渐进式采用**: 先用直接调用测试 Skills，再集成到 Agent
2. **Schema 设计**: 为 Skills 提供清晰的 input/output schema
3. **记忆管理**: 定期保存和清理记忆，避免上下文过长
4. **错误处理**: Agent 有完善的错误处理，但也需关注 Skills 的异常
5. **监控**: 观察 Agent 的工具选择逻辑，必要时调整 prompt

## 与现有系统集成

### 与原有的 Brain/Emotion/Work Agents 配合

LangChain Agent 可以与现有系统共存：

```python
# 方式 1: 完全迁移到 LangChain（推荐）
from opencareer.langchain_mcp import create_opencareer_agent

# 方式 2: 混合使用，LangChain 处理主对话，原 Agents 处理特定场景
# （需要自定义集成代码）
```

## 常见问题

### Q: 如何添加新的 Skill？

A: 在 `opencareer/skills/` 下创建新目录，实现继承 `BaseSkill` 的类，系统会自动加载。

### Q: Agent 不会选择正确的工具怎么办？

A: 优化工具的 description，或调整 system prompt 中的工具说明。

### Q: 如何调试工具调用？

A: 设置 `verbose=True`，可以看到 Agent 的思考过程和工具调用日志。

## 下一步

- 查看 `langchain_mcp_example.py` 了解更多用法
- 阅读 LangChain 官方文档了解 Agent 系统
- 探索现有 Skills 的实现方式

---

*如有问题，请查看项目文档或联系开发团队。*
