"""
岗位匹配模块：根据 chat_memory.json 中的用户画像，从岗位数据中匹配推荐职业。
"""
import pandas as pd
import json
import os
from typing import List, Dict, Tuple, Optional


class CareerMatcher:
    """基于用户画像与岗位数据做职业匹配"""

    # 专业名到搜索关键词的扩展映射
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

    # 匹配时各列的权重
    COLUMN_WEIGHTS = {
        "knowledge_label": 3.0,
        "technology_label": 2.5,
        "ability_label": 2.0,
        "industry_label": 1.5,
        "job_keywords": 2.0,
        "job_position": 1.0,
    }

    def __init__(self, excel_path: str, memory_path: str):
        self.excel_path = excel_path
        self.memory_path = memory_path
        self.df: Optional[pd.DataFrame] = None
        self.user_keywords: List[str] = []

    # ── 加载 ──────────────────────────────────────────────

    def load_data(self):
        """加载岗位 Excel 和用户记忆 JSON"""
        self.df = pd.read_excel(self.excel_path)
        self.user_keywords = self._extract_user_keywords()

    def _load_user_profile(self) -> dict:
        if not os.path.exists(self.memory_path):
            return {}
        with open(self.memory_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_user_keywords(self) -> List[str]:
        """从 chat_memory.json 提取可匹配的关键词"""
        profile = self._load_user_profile()
        if not profile:
            return []

        keywords = set()

        # 1. 用户信息 → 扩展为专业关键词
        for info in profile.get("user_info", {}).values():
            for major, words in self.MAJOR_EXPANSION.items():
                if major in str(info):
                    keywords.update(words)

        # 2. 偏好
        for pref in profile.get("preferences", {}).values():
            keywords.add(str(pref))

        # 3. 目标
        for goal in profile.get("goals", {}).values():
            keywords.add(str(goal))

        return list(keywords)

    # ── 匹配 ──────────────────────────────────────────────

    def match(self, top_n: int = 10) -> pd.DataFrame:
        """
        执行匹配，返回 top_n 条推荐岗位。
        返回 DataFrame 包含得分及原始字段。
        """
        if self.df is None:
            self.load_data()

        if not self.user_keywords:
            raise ValueError("用户画像为空，无法匹配。请先在 chat_memory.json 中填写信息。")

        df = self.df.copy()
        df["_score"] = df.apply(self._score_row, axis=1)

        result = df[df["_score"] > 0].nlargest(top_n, "_score").copy()

        # 去重: 同一公司同一岗位只保留得分最高的一条
        result = result.drop_duplicates(
            subset=["job_position", "company_name"], keep="first"
        ).head(top_n)

        result["排名"] = range(1, len(result) + 1)
        result = result.rename(columns={"_score": "匹配得分"})

        # 清洗列: JSON 数组列转可读字符串
        for col in ["knowledge_label", "technology_label", "ability_label",
                     "industry_label", "level_label"]:
            if col in result.columns:
                result[col] = result[col].apply(self._flatten_label)

        return result[[
            "排名", "匹配得分",
            "job_position", "job_keywords",
            "company_name", "main_industry",
            "knowledge_label", "technology_label", "ability_label",
            "industry_label", "level_label",
            "minimum_monthly_salary", "maximum_monthly_salary",
            "job_loc_prov", "job_loc_city",
            "education_requirement_mini", "experience_mini",
        ]]

    def _score_row(self, row) -> float:
        """对单条岗位记录打分"""
        score = 0.0

        def parse_json(val) -> List[str]:
            if pd.isna(val) or val == "\\N":
                return []
            if isinstance(val, list):
                return val
            try:
                return json.loads(str(val))
            except (json.JSONDecodeError, TypeError):
                return []

        for col, weight in self.COLUMN_WEIGHTS.items():
            if col not in row.index:
                continue
            items = parse_json(row[col])
            # 对于非 JSON 列 (job_keywords, job_position) 按逗号拆分
            if not items and col in ("job_keywords", "job_position"):
                items = str(row[col]).replace("，", ",").split(",") if not pd.isna(row[col]) else []
            items = [s.strip().lower() for s in items if s and s.strip() != "\\n"]
            user_lower = [k.lower() for k in self.user_keywords]
            hits = sum(1 for item in items if any(uk in item for uk in user_lower))
            score += hits * weight

        return score

    @staticmethod
    def _flatten_label(val) -> str:
        """将 JSON 数组标签列转为可读字符串"""
        if pd.isna(val) or val == "\\N":
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

    # ── 便捷函数 ──────────────────────────────────────────

    def print_recommendations(self, top_n: int = 10):
        """终端友好打印推荐结果"""
        try:
            result = self.match(top_n)
        except ValueError as e:
            print(f"[错误] {e}")
            return

        print("=" * 60)
        print(f"岗位推荐 (共匹配 {len(result)} 条)")
        print("=" * 60)

        for _, row in result.iterrows():
            salary = f"{row['minimum_monthly_salary']:,} ~ {row['maximum_monthly_salary']:,}"
            print(f"\n#{row['排名']}  匹配得分: {row['匹配得分']:.1f}")
            print(f"  岗位: {row['job_position']}")
            print(f"  公司: {row['company_name']}")
            print(f"  行业: {row['main_industry']}")
            print(f"  城市: {row['job_loc_prov']} {row['job_loc_city']}")
            print(f"  薪资: {salary}")
            print(f"  等级: {row['level_label']}")
            print(f"  学历要求: {row['education_requirement_mini']}")
            print(f"  经验要求: {row['experience_mini']}")

    def to_excel(self, output_path: str = "recommendations.xlsx", top_n: int = 20):
        """导出推荐结果到 Excel"""
        result = self.match(top_n)
        result.to_excel(output_path, index=False, engine="openpyxl")
        print(f"推荐结果已导出到: {output_path}")


# ── 独立函数入口 ──────────────────────────────────────────


def match_career(
    excel_path: str = "岗位匹配数据.xlsx",
    memory_path: str = "chat_memory.json",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    根据用户画像匹配岗位。

    参数:
        excel_path: 岗位数据 Excel 路径
        memory_path: 用户记忆 JSON 路径
        top_n: 返回 top N 条推荐

    返回:
        包含排名、得分和岗位信息的 DataFrame
    """
    matcher = CareerMatcher(excel_path, memory_path)
    matcher.load_data()
    return matcher.match(top_n)


def print_recommendations(
    excel_path: str = "岗位匹配数据.xlsx",
    memory_path: str = "chat_memory.json",
    top_n: int = 10,
):
    """终端打印岗位推荐"""
    matcher = CareerMatcher(excel_path, memory_path)
    matcher.load_data()
    matcher.print_recommendations(top_n)


# ── 命令行测试入口 ────────────────────────────────────────

if __name__ == "__main__":
    print_recommendations()
