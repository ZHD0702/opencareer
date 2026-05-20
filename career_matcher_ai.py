"""
岗位匹配模块 (DeepSeek 决策版)：
先关键词粗筛 → 再用 DeepSeek 分析用户画像与候选岗位，给出推荐决策。
"""
import pandas as pd
import json
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ── 辅助函数 ──────────────────────────────────────────────────

def _safe(row, col, default="未知"):
    val = row.get(col) if hasattr(row, "get") else row[col]
    if pd.isna(val) or str(val).strip() in ("", "\\N", "nan"):
        return default
    return str(val)


def _flatten_label(val) -> str:
    if pd.isna(val) or str(val).strip() in ("", "\\N"):
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    try:
        parsed = json.loads(str(val))
        if isinstance(parsed, list):
            return ", ".join(str(v) for v in parsed)
    except (json.JSONDecodeError, TypeError):
        pass
    return str(val)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ── 主类 ──────────────────────────────────────────────────────

class AICareerMatcher:
    """关键词粗筛 + DeepSeek 精排的岗位匹配器"""

    MAJOR_EXPANSION: Dict[str, List[str]] = {
        "计算机": [
            "计算机", "软件", "编程", "开发", "算法", "人工智能",
            "数据分析", "后端", "前端", "测试", "运维", "网络",
            "Java", "Python", "C++", "机器学习", "深度学习",
            "信息系统", "数据库", "IT", "信息管理",
        ],
        "电子": ["电子", "电路", "嵌入式", "硬件", "芯片", "半导体"],
        "通信": ["通信", "网络", "5G", "光纤", "无线"],
        "自动化": ["自动化", "控制", "PLC", "机器人", "传感器"],
        "机械": ["机械", "结构", "CAD", "制造", "工艺"],
    }

    COLUMN_WEIGHTS = {
        "knowledge_label": 3.0,
        "technology_label": 2.5,
        "ability_label": 2.0,
        "industry_label": 1.5,
        "job_keywords": 2.0,
        "job_position": 1.0,
    }

    def __init__(self, excel_path: str = "岗位匹配数据.xlsx", memory_path: str = "chat_memory.json"):
        self.excel_path = excel_path
        self.memory_path = memory_path
        self.df: Optional[pd.DataFrame] = None
        self.user_keywords: List[str] = []
        self.user_profile: dict = {}
        self.llm: Optional[ChatOpenAI] = None

    # ── 加载 ──────────────────────────────────────────────────

    def load_data(self, api_key: str = None, base_url: str = "https://api.deepseek.com"):
        """加载 Excel、用户画像，初始化 DeepSeek"""
        self.df = pd.read_excel(self.excel_path)
        self.user_profile = self._load_user_profile()
        self.user_keywords = self._extract_user_keywords()

        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.llm = ChatOpenAI(
            temperature=0.3,
            model="deepseek-chat",
            api_key=key,
            base_url=base_url,
        )

    def _load_user_profile(self) -> dict:
        if not os.path.exists(self.memory_path):
            return {}
        with open(self.memory_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_user_keywords(self) -> List[str]:
        profile = self.user_profile
        if not profile:
            return []
        keywords = set()
        for info in profile.get("user_info", {}).values():
            for major, words in self.MAJOR_EXPANSION.items():
                if major in str(info):
                    keywords.update(words)
        for pref in profile.get("preferences", {}).values():
            keywords.add(str(pref))
        for goal in profile.get("goals", {}).values():
            keywords.add(str(goal))
        return list(keywords)

    # ── 粗筛 ──────────────────────────────────────────────────

    def _coarse_filter(self, top_n: int = 50) -> pd.DataFrame:
        """关键词打分，截取 top_n 候选"""
        df = self.df.copy()
        df["_score"] = df.apply(self._score_row, axis=1)
        candidates = df[df["_score"] > 0].nlargest(top_n, "_score")
        candidates = candidates.drop_duplicates(
            subset=["job_position", "company_name"], keep="first"
        )
        return candidates

    def _score_row(self, row) -> float:
        score = 0.0

        def parse(val) -> List[str]:
            if pd.isna(val) or val == "\\N":
                return []
            if isinstance(val, list):
                return val
            try:
                return json.loads(str(val))
            except (json.JSONDecodeError, TypeError):
                return []

        for col, w in self.COLUMN_WEIGHTS.items():
            if col not in row.index:
                continue
            items = parse(row[col])
            if not items and col in ("job_keywords", "job_position"):
                items = str(row[col]).replace("，", ",").split(",") if not pd.isna(row[col]) else []
            items = [s.strip().lower() for s in items if s and s.strip() != "\\n"]
            ul = [k.lower() for k in self.user_keywords]
            score += sum(1 for it in items if any(uk in it for uk in ul)) * w
        return score

    # ── 用户画像文本 ──────────────────────────────────────────

    def _format_user_profile(self) -> str:
        """把 chat_memory.json 整理成可读文本"""
        p = self.user_profile
        parts = []

        if p.get("user_info"):
            parts.append("个人信息: " + "; ".join(p["user_info"].values()))
        if p.get("preferences"):
            parts.append("偏好: " + "; ".join(p["preferences"].values()))
        if p.get("important_events"):
            events = [e["content"] for e in p["important_events"]]
            parts.append("近期重要事件: " + "; ".join(events))
        if p.get("emotions"):
            parts.append("情感状态: " + "; ".join(p["emotions"].values()))
        if p.get("goals"):
            parts.append("目标: " + "; ".join(p["goals"].values()))

        return "\n".join(parts) if parts else "暂无用户画像信息"

    # ── 候选岗位文本 ──────────────────────────────────────────

    @staticmethod
    def _format_candidates(df: pd.DataFrame) -> str:
        """把候选岗位 DataFrame 格式化为 LLM 可读文本"""
        lines = []
        for i, (_, row) in enumerate(df.iterrows(), 1):
            lines.append(
                f"[{i}] 岗位: {_safe(row, 'job_position')}\n"
                f"    公司: {_safe(row, 'company_name')}\n"
                f"    行业: {_safe(row, 'main_industry')}\n"
                f"    城市: {_safe(row, 'job_loc_prov')} {_safe(row, 'job_loc_city')}\n"
                f"    月薪: {_safe(row, 'minimum_monthly_salary')} ~ {_safe(row, 'maximum_monthly_salary')}\n"
                f"    知识要求: {_flatten_label(row['knowledge_label'] if 'knowledge_label' in row.index else '')}\n"
                f"    技术要求: {_flatten_label(row['technology_label'] if 'technology_label' in row.index else '')}\n"
                f"    能力要求: {_flatten_label(row['ability_label'] if 'ability_label' in row.index else '')}\n"
                f"    行业标签: {_flatten_label(row['industry_label'] if 'industry_label' in row.index else '')}\n"
                f"    职级: {_flatten_label(row['level_label'] if 'level_label' in row.index else '')}\n"
                f"    学历要求: {_safe(row, 'education_requirement_mini')}  经验要求: {_safe(row, 'experience_mini')} 年\n"
                f"    岗位职责: {_truncate(_safe(row, 'job_responsibilities'), 200)}\n"
                f"    任职资格: {_truncate(_safe(row, 'qualifications'), 200)}"
            )
        return "\n\n".join(lines)

    # ── DeepSeek 决策 ─────────────────────────────────────────

    def match(self, coarse_top_n: int = 30, final_top_n: int = 5) -> str:
        """
        执行两阶段匹配：
        1. 关键词粗筛 coarse_top_n 个候选
        2. DeepSeek 分析、排序、推荐 final_top_n 个
        返回 DeepSeek 的推荐报告文本。
        """
        if self.df is None or self.llm is None:
            self.load_data()

        if not self.user_keywords:
            return "[错误] 用户画像为空，请先在 chat_memory.json 中填写信息。"

        # Phase 1: 粗筛
        candidates = self._coarse_filter(coarse_top_n)
        if candidates.empty:
            return "[提示] 未找到匹配的岗位，请丰富用户画像信息。"

        user_text = self._format_user_profile()
        candidate_text = self._format_candidates(candidates)

        # Phase 2: DeepSeek 决策
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位资深的职业规划顾问。你的任务是根据用户画像，从候选岗位中选出最匹配的 {final_top_n} 个岗位，并给出推荐理由。

## 分析要点
1. 用户的专业背景是否与岗位知识/技术要求匹配
2. 用户的偏好、目标是否与行业/岗位方向一致
3. 用户的学历、经验是否符合岗位要求（可适当放宽1-2年）
4. 结合用户的情感状态和近期事件，判断适合的工作节奏和环境
5. 薪资水平是否合理
6. 城市是否合适（如果用户没有明确偏好则忽略）

## 输出格式
请按以下格式输出每个推荐：

### 推荐 1: <岗位名称>
- **匹配度**: ★★★★★ (满分5星)
- **推荐理由**: <100字以内，说明为什么这个岗位适合该用户>
- **潜在顾虑**: <50字以内，需要注意的风险点>
- **公司**: <公司名>
- **城市**: <城市>
- **薪资**: <薪资范围>

... 依次输出 {final_top_n} 个推荐 ...

## 最后的总结
用2-3句话总结推荐方向和建议。"""),
            ("human", """## 用户画像
{user_profile}

## 候选岗位（共 {candidate_count} 个）
{candidates}

请分析并推荐最匹配的 {final_top_n} 个岗位。"""),
        ])

        chain = prompt | self.llm | StrOutputParser()

        report = chain.invoke({
            "user_profile": user_text,
            "candidates": candidate_text,
            "candidate_count": len(candidates),
            "final_top_n": final_top_n,
        })

        return report


# ── 独立函数入口 ──────────────────────────────────────────────


def ai_match_career(
    excel_path: str = "岗位匹配数据.xlsx",
    memory_path: str = "chat_memory.json",
    coarse_n: int = 30,
    final_n: int = 5,
    api_key: str = None,
) -> str:
    """
    DeepSeek 决策版岗位匹配。

    参数:
        excel_path: 岗位 Excel
        memory_path: 用户记忆 JSON
        coarse_n: 粗筛候选数 (推荐 20-50)
        final_n: 最终推荐数 (推荐 3-10)
        api_key: DeepSeek key，不传则读环境变量

    返回:
        DeepSeek 生成的推荐报告文本
    """
    matcher = AICareerMatcher(excel_path, memory_path)
    matcher.load_data(api_key=api_key)
    return matcher.match(coarse_top_n=coarse_n, final_top_n=final_n)


# ── 命令行入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    print("正在加载数据并分析...\n")
    report = ai_match_career()
    print(report)
