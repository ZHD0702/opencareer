# OpenCareer 简历对话引擎设计文档

## 1. 设计目标

简历对话引擎的目标不是简单地“聊完生成一份简历”，而是在多轮对话过程中持续维护一份结构化的 `ResumeState`。

每一轮用户输入都会被处理为四类结果：

1. 更新结构化简历字段。
2. 生成或优化简历片段预览。
3. 发现缺失、模糊或冲突的信息。
4. 决定下一轮最应该追问什么。

因此它更像一个“简历采集与建模引擎”，而不是一次性文本生成器。

## 2. 核心特点

### 2.1 多轮引导式对话

引擎采用漏斗式提问，将简历采集拆成几个阶段：

- `basic_info`：基础信息与求职目标。
- `experience_discovery`：发现可写入简历的项目、实习或工作经历。
- `experience_deepening`：用 STAR 方式深挖背景、任务、动作、结果。
- `soft_skill_assessment`：通过选择题收集软技能证据。
- `resume_preview`：进入简历片段预览和后续润色。

实现位置：

- `ResumeBuilderService._update_stage_and_completion`
- `ResumeBuilderService._plan_questions`
- `DEFAULT_RESUME_STATE.stage`

阶段不是靠大模型自由发挥，而是根据结构化字段完成度判断。这样对话不会轻易跑偏。

### 2.2 规则稳定 + LLM 补强

引擎使用双层抽取机制：

第一层是确定性规则抽取，用于保证基础功能稳定可控：

- 岗位、薪资、城市。
- 技能关键词。
- 经历触发词。
- 量化数字。
- 行业关键词。

第二层是 LLM extractor，用于处理规则难覆盖的问题：

- 模糊表达理解。
- STAR 深挖。
- 行业化关键词和指标建议。
- 更自然的简历 bullet。
- 更有针对性的追问。

实现位置：

- 规则层：`_extract_basics`、`_extract_target`、`_extract_skills`、`_extract_experience`
- LLM 层：`_run_llm_extractor`、`_build_llm_system_prompt`、`_apply_llm_result`
- 入口：`update_from_user_message_async`

如果 LLM 没配置、调用失败或置信度过低，引擎会回退到规则结果，不会阻塞主流程。

### 2.3 实时简历片段预览

用户描述经历后，引擎会立即尝试生成一条简历 bullet。

例如用户说：

```text
我做过招聘，一年招了80个人，负责简历筛选和面试协调
```

规则层会生成：

```text
负责招聘相关工作，完成80个招聘交付。
```

LLM 层可进一步补强为更完整的 STAR 式表达。

实现位置：

- 规则生成：`_build_preview_bullet`
- LLM 补强：`_upsert_llm_experience`
- 状态字段：`ResumeState.previews`
- 前端展示：`ResumeBuilderPanel`

### 2.4 STAR 法则深挖

每段经历会被建模为：

```json
{
  "star": {
    "situation": null,
    "task": "招聘交付",
    "action": "用户原始描述或 LLM 提炼动作",
    "result": "80人"
  }
}
```

规则层会先从关键词和数字中推断 `task/action/result`。

LLM 层会补充更细的：

- 背景是什么。
- 目标是什么。
- 用户做了什么。
- 最后结果如何。
- 哪些结果还需要确认。

实现位置：

- 规则推断：`_infer_task`、`_infer_action`、`_infer_result`
- LLM 更新：`_upsert_llm_experience`

### 2.5 模糊表达识别与追问

用户经常会说：

- “做过招聘”
- “提升了效率”
- “优化过流程”
- “负责项目”

这些表达不能直接写进简历，需要追问量级、范围和结果。

引擎通过 `FUZZY_TRIGGERS` 做规则追问，同时允许 LLM 返回 `follow_up_questions`。

实现位置：

- 规则追问：`FUZZY_TRIGGERS`
- 追问计划：`_plan_questions`
- LLM 追问字段：`follow_up_questions`
- 前端展示：`ResumeBuilderPanel` 的“下一步可以问”

### 2.6 行业特化

引擎内置行业配置：

- 互联网：转化率、留存率、DAU、GMV、效率。
- 制造业：成本、良率、产能、交付周期、损耗率。
- 应届生：课程项目、竞赛成果、实习产出、排名。

规则层通过关键词识别行业，LLM 层可以补充更细的行业建议。

实现位置：

- 行业配置：`INDUSTRY_KEYWORDS`
- 行业归一化：`_normalize_industry`
- 指标补充：`_apply_industry_specialization`
- 状态字段：`industry_insights`
- 前端展示：`ResumeBuilderPanel` 的“行业优先指标”

### 2.7 冲突检测

当前版本实现了基础冲突检测：

- 学生身份与全职工作经历冲突。

当检测到冲突时，不直接报错，而是生成澄清问题：

```text
你看起来还是学生身份，但经历里出现了全职工作。它是实习、兼职，还是正式工作？
```

实现位置：

- `ResumeBuilderService._detect_conflicts`
- `ResumeState.conflicts`
- 前端冲突提示：`ResumeBuilderPanel`

## 3. 数据模型

核心状态为 `ResumeState`，持久化在 SQLite 的 `resume_states` 表中。

表结构：

```sql
CREATE TABLE IF NOT EXISTS resume_states (
    session_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
)
```

主要字段：

```json
{
  "basics": {},
  "target": {},
  "education": [],
  "experiences": [],
  "projects": [],
  "skills": {
    "hard": [],
    "soft": []
  },
  "preferences": {},
  "previews": [],
  "industry_insights": {},
  "llm_insights": {},
  "conflicts": [],
  "unresolved_questions": [],
  "stage": "basic_info",
  "completion": 0
}
```

读写实现：

- `db.crud.get_resume_state`
- `db.crud.save_resume_state`
- `ResumeBuilderService.load_state`
- `ResumeBuilderService.apply_manual_update`

## 4. 对话处理流程

用户发消息后，聊天接口会执行：

```text
用户消息
  -> 保存 message
  -> 情绪识别
  -> ResumeBuilderService.update_from_user_message_async
  -> SSE 推送 resume_update
  -> CareerAgent / SimpleAgent 回复
  -> 如果有简历片段或追问，追加到聊天回复
  -> 保存 AI 回复
```

实现位置：

- `api/routes/chat.py`
- `ResumeBuilderService.update_from_user_message_async`
- `_build_resume_chat_hint`

前端通过 SSE 接收：

```json
{
  "type": "resume_update",
  "data": {
    "stage": "experience_deepening",
    "completion": 72,
    "latest_preview": {},
    "next_questions": [],
    "conflicts": []
  }
}
```

实现位置：

- `useSSEChat.ts`
- `chatStore.ts`
- `useResume.ts`

## 5. 前端展示

左侧“简历”栏新增了简历采集面板，展示：

- 当前采集阶段。
- 简历完成度。
- 实时片段预览。
- 技能标签。
- 行业优先指标。
- LLM 补强状态和置信度。
- 下一步追问。
- 冲突提示。

实现位置：

- `front/src/components/ResumeTab.tsx`
- `ResumeBuilderPanel`

该面板不是纯展示组件，它反映的是后端 `ResumeState` 的实时状态。

## 6. LLM Extractor 输出协议

LLM extractor 只允许返回 JSON。

核心字段：

```json
{
  "target": {},
  "skills": {},
  "experience": {
    "should_create_or_update": true,
    "type": "project",
    "star": {},
    "metrics": [],
    "bullet": "",
    "needs_confirmation": true
  },
  "industry_insights": {},
  "follow_up_questions": [],
  "conflicts": [],
  "confidence": 0.82,
  "reason": ""
}
```

应用规则：

- `confidence < 0.45` 时不应用 LLM 结果。
- LLM 不允许编造数字。
- 没有用户确认的数据必须标记 `needs_confirmation`。
- LLM 失败不影响规则抽取结果。

实现位置：

- `_build_llm_system_prompt`
- `_build_llm_user_prompt`
- `_parse_llm_json`
- `_apply_llm_result`

## 7. 当前能力边界

已经实现：

- 多轮状态化采集。
- 规则 + LLM 双层抽取。
- STAR 初步建模。
- 简历片段实时预览。
- 行业指标建议。
- 基础冲突检测。
- 前端实时展示。

尚未完整实现：

- 多段经历的智能合并与去重。
- 完整教育时间与工作时间冲突检测。
- 真正的自适应排版。
- 多行业完整配置库。
- 简历最终导出模板联动。
- LLM extractor 的专门单元测试与评估集。

## 8. 后续扩展建议

优先级建议如下：

1. 建立抽取评估集，覆盖招聘、技术项目、运营增长、制造降本、应届生项目。
2. 给每段经历增加 `quality_score`，判断是否具备可写入简历的完整度。
3. 增加“确认机制”，让用户确认 LLM 推断出的 bullet 和指标。
4. 把行业配置迁移成独立 JSON/YAML，便于扩展。
5. 将 `ResumeState` 与最终简历导出模板打通。

## 9. 关键文件索引

- `web/backend/services/resume_builder_service.py`
- `web/backend/db/crud.py`
- `web/backend/services/analysis_service.py`
- `web/backend/api/routes/chat.py`
- `web/backend/api/routes/resume.py`
- `web/backend/api/schemas.py`
- `web/front/src/components/ResumeTab.tsx`
- `web/front/src/hooks/useResume.ts`
- `web/front/src/hooks/useSSEChat.ts`
- `web/front/src/stores/chatStore.ts`
