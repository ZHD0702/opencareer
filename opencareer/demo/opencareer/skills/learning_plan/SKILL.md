---
name: learning-plan
description: "Analyze skill gaps between user's current skills and career goals, then generate a personalized learning plan with resource recommendations."
---

# Learning Plan

Generate a structured, time-bounded learning plan based on target role and current skill set.

## Workflow

### Step 1: Collect Information

This SKILL is called by the work_agent with the following inputs:

- **career_goal** (required): Target position, e.g. "Frontend Developer", "Data Scientist"
- **current_skills** (required): List of skills the user already possesses
- **timeframe_weeks** (optional, default 12): Range 1-52
- **learning_style** (optional): visual / auditory / reading / kinesthetic / mixed

If required fields are missing, ask the user in Chinese to provide them.

### Step 2: Analyze Skill Gap

Compare the user's current skills against the target role's requirements.

**Built-in career paths:**
- Frontend: HTML/CSS → JavaScript → Framework (React/Vue) → Build Tools → Full-stack
- Backend: Language Basics → Database → API Design → Microservices → Cloud Deployment
- Data Science: Python → Statistics → ML Algorithms → Deep Learning → MLOps
- DevOps: Linux → CI/CD → Containerization → Cloud Native → Monitoring
- Product Management: Requirements → Prototyping → Data Analysis → Project Management → Strategy

**Output the gap analysis:**
- Missing skills with priority (high/medium/low)
- Match percentage against target role
- Recommended skills to prioritize

### Step 3: Generate Weekly Learning Plan

Build a week-by-week plan based on gaps, timeframe, and learning style:

**Plan structure:**
- Phases: Foundation → Advancement → Practice
- Each week: topic, content, exercises
- Milestone checkpoints at each phase end

**Style adaptation:**
| Style | Recommended Resources |
|-------|---------------------|
| visual | Video tutorials, diagrams, mind maps |
| auditory | Podcasts, lectures, discussion groups |
| reading | Documentation, books, blogs |
| kinesthetic | Hands-on projects, coding exercises |
| mixed | Combined resources |

### Step 4: Recommend Learning Resources

Map each skill gap to learning resources:
- Free resources (docs, tutorials, open-source projects)
- Paid courses (MOOC, professional training)
- Practice projects (portfolio-worthy exercises)

### Step 5: Output and Track

Return the complete plan, record it to user profile for progress tracking.

## Output Format

用中文回复。返回以下结构：

1. **Skill Gap Analysis Report**
   - 缺失技能及优先级（高/中/低）
   - 与目标岗位匹配度百分比
2. **Structured Weekly Learning Plan**
   - 分阶段（基础期 → 提升期 → 实战期）
   - 每周主题、内容、练习任务
   - 里程碑检查点
3. **Recommended Courses and Resources List**
4. **Estimated Completion Date**
