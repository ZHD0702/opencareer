---
name: mock_interview
description: "Provide AI-driven mock interview practice with multiple interview types, real-time question generation, and detailed evaluation feedback."
---

# Mock Interview

Simulate realistic interviews with adaptive questioning and multi-dimensional assessment.

## Workflow

### Step 1: Configure Interview Parameters

This SKILL is called by the work_agent with the following inputs:

- **position** (required): Target role, e.g. "Software Engineer", "Product Manager"
- **experience_level** (required): entry / mid / senior / executive
- **interview_type** (optional, default mixed): technical / behavioral / system_design / mixed
- **question_count** (optional, default 10): Range 1-20
- **time_limit_minutes** (optional, default 60): Range 10-120

If required fields are missing, ask the user in Chinese to provide them.

### Step 2: Generate Questions

Generate questions tailored to position, level, and type:

**Question distribution (mixed mode):**
| Type | Weight | Description |
|------|--------|-------------|
| Technical | 40% | Algorithms, data structures, language/framework knowledge |
| Behavioral | 30% | Past experience, teamwork, conflict resolution, leadership |
| System Design | 30% | Architecture, scalability, trade-off analysis |

**Difficulty by level:**
- entry: Basic concepts, simple algorithms, learning ability
- mid: Project experience, medium complexity, best practices
- senior: Architecture decisions, team leadership, complex problems
- executive: Strategic thinking, cross-team coordination, business acumen

### Step 3: Conduct Q&A Session

Present questions one at a time. For each user response, record:
- Answer text
- Response time
- Whether skipped

Call `evaluate_response()` after each answer for real-time scoring (do not interrupt the flow).

### Step 4: Multi-Dimension Evaluation

Evaluate each response across four dimensions:

| Dimension | Weight | Criteria |
|-----------|--------|----------|
| Technical Knowledge | 40% | Accuracy, depth, relevance, examples |
| Communication | 30% | Clarity, structure, conciseness, confidence |
| Problem Solving | 20% | Approach, creativity, efficiency, adaptability |
| Cultural Fit | 10% | Enthusiasm, collaboration, values, growth mindset |

**Score levels:** Excellent (9-10) / Good (7-8) / Average (5-6) / Needs Improvement (3-4) / Poor (1-2)

### Step 5: Output Full Feedback

After the session ends, generate the complete evaluation report.

## Output Format

用中文回复。返回以下结构：

1. **Overall Score and Dimension Breakdown**
   - 各维度得分及总分
2. **Per-Question Detailed Feedback**
   - 每道题的优点、改进建议、参考回答
3. **Weak Area Analysis**
   - 薄弱环节及待加强技能
4. **Improvement Recommendations**
   - 学习资源链接和练习方向建议
