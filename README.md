# OpenCareer — AI 职业伴侣

基于 LangChain + MCP + DeepSeek 构建的智能职业发展助手，集情感支持、简历生成、岗位匹配于一体。

## 功能特性

- **AI 职业顾问**：基于 DeepSeek 大模型的职业咨询、面试准备、情绪疏导
- **简历生成与优化**：中文简历结构化生成、ATS 关键词检查、PDF 导出
- **智能岗位匹配**：AI 决策优先 + 本地关键词匹配回退，从 31MB 岗位数据中推荐最佳职位
- **长期记忆**：LLM 自动提取用户画像（背景、技能、目标、情绪），跨会话持久化
- **MCP 协议集成**：通过 MCP Server 暴露 `resume_skill` 工具，支持 Claude Desktop 等客户端调用
- **自动启动**：一键启动自动拉起 MCP 服务器，无需手动管理进程

## 技术架构

```
User CLI (main.py)
    │
    ▼
CareerAgent (LangChain Agent)
    │  LLM: DeepSeek Chat (via langchain-openai)
    │  Tools: MCP Server → resume_skill
    │  Memory: JSON 长期记忆 + 对话历史
    │
    ├── MCP Server (FastMCP, port 8001)
    │   └── resume_skill: generate / optimize / ats_check / export_pdf
    │
    └── Job Matcher
        ├── career_matcher_ai.py  (DeepSeek AI 决策，首选)
        └── career_matcher.py     (关键词打分，回退)
```

## 项目结构

```
├── main.py                          # CLI 入口，自动启动 MCP + 主程序
├── opencareer/
│   ├── agents/
│   │   └── career_agent.py          # LangChain Agent，工具调用 + 记忆管理
│   ├── mcp/
│   │   ├── server.py                # FastMCP 服务器
│   │   ├── tools/
│   │   │   ├── resume_tool.py       # 简历生成/优化/ATS/PDF
│   │   │   └── generate_resume_pdf.py
│   │   ├── prompts/                 # Skill 注册 + 提示词
│   │   └── skills/                  # 内置技能（简历编辑器等）
│   └── prompts/career/              # System prompt + 信息提取 prompt
├── career_matcher.py                # 本地关键词岗位匹配
├── career_matcher_ai.py             # DeepSeek AI 岗位匹配（首选）
├── career_memory.json               # 长期记忆存储（自动生成）
├── 岗位匹配数据.xlsx                 # 岗位数据源（需自行准备）
├── docs/                            # 匹配器文档
├── outputs/resume/                  # 简历 PDF 输出目录
└── .github/workflows/               # CI 流水线
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- DeepSeek API Key

### 2. 安装依赖

```bash
pip install langchain langchain-openai langchain-mcp-adapters mcp python-dotenv pandas openpyxl
```

### 3. 配置 API 密钥

在项目根目录 `.env` 文件中配置：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

### 4. 启动

```bash
# 自动启动 MCP 服务器 + 主程序
python main.py

# 流式输出模式
python main.py --stream

# 离线模式（不启动 MCP，纯 LLM 对话）
python main.py --no-mcp
```

程序会自动：
1. 检查并启动 MCP 服务器（端口 8001）
2. 连接到 MCP 服务并加载工具
3. 进入交互式对话界面

退出时自动关闭 MCP 服务器。

### 5. 可用命令

```
quit / exit  — 退出程序
memory       — 查看记忆摘要
detail       — 查看详细记忆内容
clear        — 清除所有记忆
match        — 岗位匹配推荐（优先 AI 决策，回退本地匹配）
/stream      — 切换流式输出模式
```

## 岗位匹配

输入 `match` 命令即可触发岗位匹配：

1. **AI 优先级**：DeepSeek 分析用户画像 + 岗位数据，生成个性化推荐报告
2. **本地回退**：若 AI 不可用，自动切换到关键词打分匹配

匹配数据需要准备 `岗位匹配数据.xlsx` 文件（含岗位名称、公司、行业、薪资、技能要求等列）。

## MCP 工具

### resume_skill — 简历工作流

```
generate   — 生成结构化简历草稿
optimize   — 按 JD 优化简历
ats_check  — ATS 关键词匹配检查
export_pdf — 导出 PDF
```

## License

MIT License
