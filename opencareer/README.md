# OpenCareer — AI 求职助手

OpenCareer 是一个基于多智能体架构的 AI 求职助手系统，帮助求职者在技能发展、面试准备和情感支持等方面获得全方位协助。

## 项目概览

```
OpenCareer/
├── demo/                    # 主演示项目（4智能体 + YAML Prompt 系统）
│   ├── quick_dialogue.py   # 交互式终端入口
│   └── opencareer/         # 核心代码
│       ├── agents/          # 4智能体：Brain、Work、Emotion、Log
│       ├── prompts/         # YAML Prompt 系统
│       ├── skills/          # 3核心SKILL：学习计划、模拟面试、情感支持
│       ├── mcp/             # MCP Server 框架
│       └── memory/          # 记忆系统（ChromaDB + SQLite）
├── resume-cn-career/        # 简历优化与职业发展模块
├── skills/                  # 共享技能定义与配置
├── 工作日志.md               # 项目工作日志
├── 情感需求分析模块机制.md    # 情感分析三层防护机制文档
├── 技术决策与前期规划.txt      # 技术选型与架构决策记录
├── 设计文档.txt               # 系统设计文档
├── 需求文档.txt               # 需求分析文档
└── 并发架构方案.txt            # 并发架构设计文档
```

## 核心特性

- **4智能体架构**：Brain Agent（入口调度）→ Work Agent（求职帮助）/ Emotion Agent（情感支持）/ Log Agent（信息提取）
- **Judge-and-Handoff 模式**：每轮对话重新分析用户需求，按优先级路由到最合适的智能体
- **三层情绪防护**：Brain Agent 路由分析 → Work Agent support_intensity 检查 → 关键词情绪检测
- **YAML Prompt 系统**：所有提示词通过 `PromptRegistry` 统一管理，支持 persona 继承
- **3 个核心 SKILL**：学习计划、模拟面试、情感支持（通过 MCP Server 动态加载）
- **真实 AI 驱动**：基于 DeepSeek API，支持结构化 JSON 输出

## 快速开始

详见 [demo/README.md](demo/README.md) 的 Quick Start 章节。

```bash
cd demo
pip install -r requirements.txt
# 配置 .env 中的 DeepSeek API 密钥
python quick_dialogue.py
```

## 架构设计

| 组件 | 说明 |
|------|------|
| **Brain Agent** | 入口智能体，分析用户需求+情绪，按优先级路由到下游 Agent；支持跨轮次积累情绪状态路由覆盖 |
| **Work Agent** | 处理求职相关问题，路由到对应 SKILL，内置情绪检测 |
| **Emotion Agent** | 情感支持，根据 support_intensity 校准回应语气 |
| **Log Agent** | 后台信息提取（fire-and-forget） |
| **ConversationContext** | 共享对话状态：历史、情绪追踪、用户画像、跨轮次情感积累评估 |

详细架构说明见 [情感需求分析模块机制.md](情感需求分析模块机制.md)。

## 项目状态

- ✅ 核心架构验证完成
- ✅ 真实 AI 模式运行正常（DeepSeek API）
- ✅ 多智能体协作与情绪路由
- ✅ 跨轮次积累情绪状态路由覆盖
- ⚠️ MCP Server 完整实现（待启动）
- ⚠️ 记忆系统完全集成（模拟实现）
- ⚠️ Web 界面（待开发）
