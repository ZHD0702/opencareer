---
name: emotion-support
description: "Provide emotional support and encouragement for job seekers. Identify emotional states and deliver personalized responses to alleviate stress and maintain a positive mindset."
---

# Emotion Support

Recognize the user's emotional state during their job search and provide appropriate emotional support and actionable advice.

## Workflow

### Step 1: Identify Emotional State

This SKILL is called by the emotion_agent with the following inputs:

- **current_mood** (required): happy / neutral / stressed / anxious / discouraged / confident
- **recent_experience** (optional): e.g. "interview rejected", "received offer"
- **support_type** (optional): encouragement / motivation / advice / validation / listening
- **job_search_stage** (optional): starting / applying / interviewing / waiting / negotiating / accepted
- **user_id** (optional): User identifier for history tracking

If the emotional state is unclear, ask probing questions in Chinese to help the user express their feelings.

### Step 2: Match Response Strategy

Select strategy based on emotional state:

| Mood | Strategy | Core Goal |
|------|----------|-----------|
| stressed | Stress relief, breathing exercises, mindfulness | Reduce tension |
| anxious | Reduce uncertainty, step-by-step approach, restore control | Rebuild sense of safety |
| discouraged | Review past achievements, rebuild confidence, social support | Improve self-efficacy |
| neutral | Maintain balanced mindset, set clear goals | Keep stable state |
| happy/confident | Ride the momentum, expand proactively, help others | Amplify positive energy |

### Step 3: Build Emotional Response

Construct a three-layer response:

1. **Empathy & validation**: Acknowledge and normalize the user's feelings
2. **Core support message**: Encouragement/advice/validation based on strategy
3. **Actionable suggestions**: Specific steps (e.g. "Take 5 deep breaths", "List 3 achievements")

Also generate:
- Mood improvement tips (3-5 concrete actions)
- Recommended break time (based on stress level)
- Follow-up questions to deepen the conversation

### Step 4: Track Mood Changes

Record this interaction to user profile:
- Update mood history timeline
- Detect trends (e.g. multiple consecutive low days → suggest stronger intervention)
- Proactive check-in for low-frequency users

### Step 5: Output Result

Return structured emotional support information.

## Output Format

用中文回复。返回以下结构：

1. **Emotional Response**
   - 共情消息 + 肯定语句 + 建议行动 + 推荐资源
2. **Mood Improvement Tips**
   - 3-5 条具体的情绪改善行动建议
3. **Follow-up Questions**
   - 1-3 个促进深入交流的问题
4. **Recommended Break Time**
   - 建议休息时间（分钟）
5. **Mood Tracking Info**
   - 情绪变化趋势，连续状态提醒

## Important Notes

- 情感支持不能替代专业心理咨询
- 如果检测到用户有持续严重情绪低落或自我伤害倾向，建议寻求专业帮助
- 建议结合对话上下文和用户历史更精准地识别情绪
