# career_matcher — 关键词打分版岗位匹配

## 概述

`career_matcher.py` 是一个基于关键词加权打分的岗位匹配模块。它读取 `chat_memory.json` 中的用户画像和 `岗位匹配数据.xlsx` 中的 5 万条岗位数据，通过多字段关键词匹配计算每条岗位的适配得分，最终返回 Top-N 推荐结果。

## 适用场景

- 快速批量筛选：几秒内完成 5 万条岗位打分
- 不需要 API 调用，离线可用
- 用户画像较丰富时效果更好

## 匹配流程

```
chat_memory.json                    岗位匹配数据.xlsx (50,000 条)
      │                                       │
      ▼                                       │
  提取关键词 ◄─────────────────────────────────┘
  (专业扩展映射)                               │
      │                                       │
      ▼                                       ▼
  用户关键词集合                     逐条计算匹配得分
  (如"计算机"→20个词)                ┌─────────────────┐
                                    │ knowledge_label  │ x3.0
                                    │ technology_label │ x2.5
                                    │ ability_label    │ x2.0
                                    │ job_keywords     │ x2.0
                                    │ industry_label   │ x1.5
                                    │ job_position     │ x1.0
                                    └─────────────────┘
                                            │
                                     去重 → 排序 → Top-N
                                            │
                                            ▼
                                      DataFrame / Excel
```

## 核心模块

### CareerMatcher 类

| 方法 | 说明 |
|---|---|
| `load_data()` | 加载 Excel 和用户 JSON |
| `match(top_n=10)` | 执行匹配，返回带得分和排名的 DataFrame |
| `print_recommendations(top_n=10)` | 终端友好打印 |
| `to_excel(output_path, top_n=20)` | 导出推荐到 Excel |

### 独立函数

```python
from career_matcher import match_career, print_recommendations

# 返回 DataFrame
df = match_career(excel_path="岗位匹配数据.xlsx",
                  memory_path="chat_memory.json",
                  top_n=10)

# 终端打印
print_recommendations()
```

## 关键词扩展

从用户画像中的专业名称自动扩展为岗位搜索关键词：

| 专业 | 扩展关键词数 | 示例 |
|---|---|---|
| 计算机 | 20 个 | Python, Java, 算法, AI, 测试, 前端, 后端... |
| 电子 | 6 个 | 电子, 嵌入式, 硬件, 芯片... |
| 通信 | 5 个 | 通信, 5G, 光纤, 无线... |
| 自动化 | 5 个 | PLC, 机器人, 传感器... |
| 机械 | 5 个 | 结构, CAD, 制造, 工艺... |

可在 `MAJOR_EXPANSION` 字典中扩展更多专业。

## 打分权重

| 字段 | 权重 | 说明 |
|---|---|---|
| `knowledge_label` | 3.0 | 理论知识匹配（最高权重，确保专业对口） |
| `technology_label` | 2.5 | 技术栈匹配 |
| `ability_label` | 2.0 | 实操能力匹配 |
| `job_keywords` | 2.0 | 岗位关键词匹配 |
| `industry_label` | 1.5 | 行业方向匹配 |
| `job_position` | 1.0 | 岗位名称匹配 |

## 数据要求

### chat_memory.json

字段越丰富匹配越精准：

```json
{
  "user_info": {"时间戳": "计算机专业"},
  "preferences": {"时间戳": "喜欢后端开发"},
  "goals": {"时间戳": "成为架构师"},
  "important_events": [{"content": "..."}],
  "emotions": {"时间戳": "..."}
}
```

### 岗位匹配数据.xlsx

50,000 行 x 49 列，核心匹配字段：
- `knowledge_label` — JSON 数组，理论知识标签
- `technology_label` — JSON 数组，技术栈标签
- `ability_label` — JSON 数组，能力标签
- `industry_label` — JSON 数组，行业标签
- `job_keywords` — 逗号分隔的岗位关键词
- `job_position` — 岗位名称

## 输出字段

| 列名 | 说明 |
|---|---|
| 排名 | 推荐序号 |
| 匹配得分 | 加权命中总分 |
| job_position | 岗位名称 |
| job_keywords | 岗位关键词 |
| company_name | 公司名称 |
| knowledge_label | 理论知识要求 |
| technology_label | 技术要求 |
| ability_label | 能力要求 |
| industry_label | 行业标签 |
| level_label | 职级 |
| minimum_monthly_salary | 最低月薪 |
| maximum_monthly_salary | 最高月薪 |
| job_loc_prov / job_loc_city | 省市 |
| education_requirement_mini | 学历要求 |
| experience_mini | 经验要求（年） |

## 运行

```bash
python career_matcher.py
```
