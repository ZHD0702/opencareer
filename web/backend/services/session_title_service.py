import re


TITLE_RULES = (
    (("简历", "resume", "cv"), "优化简历"),
    (("模拟面试", "面试", "interview"), "面试准备"),
    (("薪资", "谈薪", "涨薪", "salary"), "薪资谈判"),
    (("职业规划", "职业发展", "规划", "转行"), "职业规划"),
    (("投递", "岗位", "找工作", "求职"), "求职策略"),
    (("焦虑", "崩溃", "难受", "压力", "受不了"), "求职情绪支持"),
    (("技能", "能力", "学习路线", "提升"), "技能提升"),
)


def generate_session_title(user_message: str) -> str:
    text = (user_message or "").strip()
    lower_text = text.lower()

    for keywords, title in TITLE_RULES:
        if any(keyword in lower_text or keyword in text for keyword in keywords):
            return title

    cleaned = re.sub(r"\s+", "", text)
    cleaned = re.sub(r"[，。！？、,.!?；;：:]+", "", cleaned)
    if not cleaned:
        return "新的会话"

    return cleaned[:14]
