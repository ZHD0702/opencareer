"""
MCP Server for OpenCareer — exposes career companion tools via MCP protocol.

Uses FastMCP with Streamable HTTP transport so langchain-mcp-adapters
can connect via MultiServerMCPClient.

Run:
    python -m opencareer.mcp.server
or:
    python opencareer/mcp/server.py
"""

import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from opencareer.mcp.prompts import (
    create_skill_registry,
    get_career_templates,
    get_emotion_templates,
    get_interview_templates,
)

logger = logging.getLogger("opencareer.mcp.server")

# ------------------------------------------------------------------
# FastMCP application
# ------------------------------------------------------------------

mcp = FastMCP(
    name="OpenCareer MCP Server",
    instructions="AI Career Companion tools: emotion support, career advice, resume analysis, interview prep",
    host="127.0.0.1",
    port=8001,
)

# ------------------------------------------------------------------
# Load prompt templates
# ------------------------------------------------------------------

_EMOTION_RESPONSES, _SUPPORT_TIPS = get_emotion_templates()
_CAREER_TOPICS, _NEXT_STEP = get_career_templates()
_COMMON_QUESTIONS, _INTERVIEW_TIPS, _STAR_REMINDER = get_interview_templates()


@mcp.tool(name="emotion_support", description="提供情感支持和情绪疏导")
async def emotion_support(
    user_input: str,
    emotion_category: Optional[str] = None,
) -> Dict[str, Any]:
    """Provide emotional support based on user input and detected emotion.

    Args:
        user_input: User's message/content
        emotion_category: Detected emotion category (焦虑/挫败/迷茫/压力)

    Returns:
        Support response with acknowledgment, guidance, and tips
    """
    # Auto-detect emotion if not provided
    if not emotion_category:
        for keyword, _ in _EMOTION_RESPONSES.items():
            if keyword != "default" and keyword in user_input:
                emotion_category = keyword
                break

    resp = _EMOTION_RESPONSES.get(emotion_category or "", _EMOTION_RESPONSES["default"])

    return {
        "emotion_detected": emotion_category or "general",
        "acknowledgment": resp["acknowledgment"],
        "guidance": resp["guidance"],
        "support_tips": _SUPPORT_TIPS[:2],
        "technique": "active_listening",
    }


# ------------------------------------------------------------------
# Tool: career_advice
# ------------------------------------------------------------------


@mcp.tool(name="career_advice", description="提供求职和职业发展建议")
async def career_advice(
    user_input: str,
    topics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Provide career advice based on detected topics.

    Args:
        user_input: User's question or message
        topics: Pre-detected career topics (简历/面试/谈薪资/职业规划)

    Returns:
        Career advice with actionable steps
    """
    if not topics:
        topics = []
        for topic_name, info in _CAREER_TOPICS.items():
            if topic_name == "default":
                continue
            for kw in info["keywords"]:
                if kw in user_input:
                    topics.append(topic_name)
                    break

    if not topics:
        topics = ["default"]

    all_advice: List[str] = []
    for topic in topics:
        info = _CAREER_TOPICS.get(topic, _CAREER_TOPICS["default"])
        all_advice.extend(info["advice"])

    return {
        "detected_topics": topics,
        "advice": all_advice[:5],
        "next_step_suggestion": _NEXT_STEP,
    }


# ------------------------------------------------------------------
# Tool: resume_analysis
# ------------------------------------------------------------------

@mcp.tool(name="resume_analysis", description="分析简历内容并提供改进建议")
async def resume_analysis(
    resume_text: str,
    target_role: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze resume content and suggest improvements.

    Args:
        resume_text: Full text of the resume
        target_role: Target job role for tailored suggestions

    Returns:
        Analysis results with scores and suggestions
    """
    issues: List[Dict[str, str]] = []
    strengths: List[str] = []
    score = 70  # base score

    # Length check
    lines = resume_text.strip().split("\n")
    # 计算简历行数用于后续分析
    line_count = len(lines)
    # 使用 line_count 进行简历长度评估
    if line_count < 10:
        issues.append({
            "type": "format",
            "severity": "low",
            "detail": "简历行数较少，建议检查是否遗漏重要内容",
        })
    if len(resume_text) < 200:
        issues.append({
            "type": "content",
            "severity": "high",
            "detail": "简历内容过短，建议至少包含教育背景、实习/项目经历、技能三个模块",
        })
        score -= 20
    elif len(resume_text) > 2000:
        issues.append({
            "type": "format",
            "severity": "medium",
            "detail": "简历过长，建议精简到1页（800-1200字）",
        })
        score -= 10

    # Keyword checks
    essential_sections = ["教育", "经历", "技能"]
    for section in essential_sections:
        if section not in resume_text:
            issues.append({
                "type": "structure",
                "severity": "high",
                "detail": f"缺少'{section}'模块，这是HR最关注的部分",
            })
            score -= 10

    # STAR method check
    star_indicators = ["负责", "参与", "完成", "提升", "降低", "实现", "优化"]
    found_star = any(w in resume_text for w in star_indicators)
    if found_star:
        strengths.append("使用了行动导向的描述词汇，有量化成果的意识")
        score += 10
    else:
        issues.append({
            "type": "wording",
            "severity": "medium",
            "detail": "建议使用更多动词开头的描述（如'负责XX项目，提升了XX%'），并量化成果",
        })

    # Number/metrics check
    import re
    has_numbers = bool(re.search(r"\d+%|\d+倍|\d+人|\d+万|\d+个", resume_text))
    if has_numbers:
        strengths.append("包含量化数据，增强了说服力")
        score += 10
    else:
        issues.append({
            "type": "content",
            "severity": "medium",
            "detail": "建议添加量化成果（如'提升效率30%'、'管理5人团队'）",
        })

    score = max(0, min(100, score))

    return {
        "score": score,
        "grade": "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D",
        "strengths": strengths,
        "issues": issues,
        "target_role": target_role,
        "summary": f"简历整体评分 {score}/100。{'表现不错，' if score >= 70 else '有较大提升空间，'}共发现{len(issues)}个可改进项。",
    }


# ------------------------------------------------------------------
# Tool: interview_prep
# ------------------------------------------------------------------


@mcp.tool(name="interview_prep", description="提供面试准备建议和模拟面试问题")
async def interview_prep(
    role: Optional[str] = None,
    question_type: str = "通用",
) -> Dict[str, Any]:
    """Provide interview preparation guidance and mock questions.

    Args:
        role: Target job role for tailored questions
        question_type: Type of interview (通用/技术/行为)

    Returns:
        Interview prep with common questions and tips
    """
    questions = _COMMON_QUESTIONS.get(question_type, _COMMON_QUESTIONS["通用"])

    return {
        "role": role or "通用",
        "question_type": question_type,
        "common_questions": questions,
        "preparation_tips": _INTERVIEW_TIPS,
        "star_reminder": _STAR_REMINDER,
    }


# ------------------------------------------------------------------
# Auto-register Skills as MCP Tools
# ------------------------------------------------------------------

create_skill_registry(
    mcp,
    tool_catalog={
        "emotion_support": emotion_support,
        "career_advice": career_advice,
        "resume_analysis": resume_analysis,
        "interview_prep": interview_prep,
    },
)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def main():
    """Start the MCP server."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("Starting OpenCareer MCP Server...")
    logger.info("Tools available: emotion_support, career_advice, resume_analysis, interview_prep")

    # Run with Streamable HTTP transport
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
