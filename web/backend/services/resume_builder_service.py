from __future__ import annotations

import re
import json
from copy import deepcopy
from datetime import datetime
from typing import Any

from db.crud import get_resume_state, save_resume_state


DEFAULT_RESUME_STATE: dict[str, Any] = {
    "basics": {
        "name": None,
        "grade_level": None,
        "major": None,
        "school": None,
    },
    "target": {
        "role": None,
        "industry": None,
        "city": None,
        "salary_expectation": None,
        "job_search_stage": "exploring",
    },
    "education": [],
    "experiences": [],
    "projects": [],
    "skills": {
        "hard": [],
        "soft": [],
    },
    "preferences": {
        "work_style": None,
        "pressure_response": None,
    },
    "previews": [],
    "industry_insights": {
        "keywords": [],
        "preferred_metrics": [],
        "notes": [],
    },
    "llm_insights": {
        "last_applied": False,
        "confidence": 0.0,
        "reason": None,
    },
    "conflicts": [],
    "unresolved_questions": [],
    "stage": "basic_info",
    "completion": 0,
    "last_update_summary": None,
}


class ResumeBuilderService:
    """Deterministic MVP for guided resume-state extraction."""

    ROLE_PATTERNS = (
        re.compile(r"(?:想投|想找|想做|应聘|投递)([\w\u4e00-\u9fffA-Za-z +#/-]{2,18})(?:岗位|职位|方向)"),
        re.compile(r"(?:目标|求职意向|想投|投递|应聘|找)(?:的)?(?:岗位|职位|方向)?[是为叫：:\s]*([\w\u4e00-\u9fffA-Za-z +#/-]{2,24})"),
        re.compile(r"([\w\u4e00-\u9fffA-Za-z +#/-]{2,24})(?:岗位|职位|方向)"),
    )
    SALARY_PATTERN = re.compile(r"(\d+(?:\.\d+)?\s*[kK万]?\s*[-~到至]\s*\d+(?:\.\d+)?\s*[kK万]?|\d+\s*[kK万]\+?)")
    CITY_PATTERN = re.compile(r"(北京|上海|广州|深圳|杭州|成都|南京|武汉|西安|苏州|天津|重庆|厦门|长沙|郑州|青岛)")
    NUMBER_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(?:\s*)(人天|人|个|次|%|％|万|k|K|天|周|月|年)?")

    FUZZY_TRIGGERS = {
        "招聘": "你负责的招聘一年大概完成了多少人？有没有关键岗位或平均到岗周期？",
        "提升效率": "这个效率大概提升了多少？如果没有精确数字，是节省时间、人力，还是缩短流程？",
        "优化流程": "优化前后最大的变化是什么？能用周期、成本或错误率描述一下吗？",
        "负责项目": "这个项目当时的目标是什么？你具体负责哪一块，最后结果怎么样？",
        "做过项目": "这个项目可以按背景、动作、结果拆一下吗？你最关键的贡献是什么？",
        "管理": "你管理的是人、流程、供应商还是项目？规模大概多大？",
    }

    INDUSTRY_KEYWORDS = {
        "internet": {
            "markers": ("互联网", "增长", "用户", "转化", "留存", "迭代", "产品", "数据"),
            "label": "互联网",
            "preferred_metrics": ["转化率", "留存率", "DAU", "GMV", "效率"],
        },
        "manufacturing": {
            "markers": ("制造", "工厂", "产线", "精益", "良率", "降本", "供应链", "质量"),
            "label": "制造业",
            "preferred_metrics": ["成本", "良率", "产能", "交付周期", "损耗率"],
        },
        "fresh_graduate": {
            "markers": ("应届", "大三", "大四", "研一", "研二", "课程", "竞赛", "实习"),
            "label": "应届生",
            "preferred_metrics": ["课程项目", "竞赛成果", "实习产出", "排名"],
        },
    }

    HARD_SKILLS = (
        "Python", "Java", "C++", "Go", "React", "Vue", "SQL", "Excel", "Docker",
        "Kubernetes", "Linux", "数据分析", "招聘", "面试", "人才画像", "活动策划",
        "用户增长", "精益管理", "供应链", "成本控制",
    )
    SOFT_SKILLS = ("沟通", "抗压", "推进", "协调", "复盘", "学习", "领导", "执行")

    def load_state(self, session_id: str) -> dict[str, Any]:
        state = get_resume_state(session_id)
        if not state:
            return deepcopy(DEFAULT_RESUME_STATE)
        merged = deepcopy(DEFAULT_RESUME_STATE)
        self._deep_update(merged, state)
        return merged

    def update_from_user_message(self, session_id: str, user_message: str) -> dict[str, Any]:
        state = self.load_state(session_id)
        text = (user_message or "").strip()
        if not text:
            return self._build_event(session_id, state, [])

        changes: list[str] = []
        self._extract_basics(text, state, changes)
        self._extract_target(text, state, changes)
        self._extract_skills(text, state, changes)
        self._extract_experience(text, state, changes)
        self._extract_preferences(text, state, changes)
        self._detect_conflicts(state)
        self._plan_questions(text, state)
        self._update_stage_and_completion(state)

        state["last_update_summary"] = "；".join(changes) if changes else "记录了新的对话线索"
        save_resume_state(session_id, state)
        return self._build_event(session_id, state, changes)

    async def update_from_user_message_async(self, session_id: str, user_message: str) -> dict[str, Any]:
        state = self.load_state(session_id)
        text = (user_message or "").strip()
        if not text:
            return self._build_event(session_id, state, [])

        changes: list[str] = []
        self._extract_basics(text, state, changes)
        self._extract_target(text, state, changes)
        self._extract_skills(text, state, changes)
        self._extract_experience(text, state, changes)
        self._extract_preferences(text, state, changes)

        llm_result = await self._run_llm_extractor(text, state)
        if llm_result:
            self._apply_llm_result(llm_result, state, changes)

        self._apply_industry_specialization(state, changes)
        self._detect_conflicts(state)
        self._plan_questions(text, state, llm_result)
        self._update_stage_and_completion(state)

        state["last_update_summary"] = "；".join(changes) if changes else "记录了新的对话线索"
        save_resume_state(session_id, state)
        return self._build_event(session_id, state, changes)

    def apply_manual_update(self, session_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        state = self.load_state(session_id)
        basics = state["basics"]
        target = state["target"]

        for key in ("grade_level", "major", "school"):
            if key in fields:
                basics[key] = fields.get(key)
        if "target_role" in fields:
            target["role"] = fields.get("target_role")
        if "job_search_stage" in fields:
            target["job_search_stage"] = fields.get("job_search_stage") or "exploring"
        if "skill_focus" in fields and fields.get("skill_focus") is not None:
            state["skills"]["hard"] = self._unique(fields["skill_focus"])
        if "common_concerns" in fields and fields.get("common_concerns") is not None:
            state["unresolved_questions"] = fields["common_concerns"]
        if "background_summary" in fields:
            state["last_update_summary"] = fields.get("background_summary")

        self._update_stage_and_completion(state)
        save_resume_state(session_id, state)
        return state

    def to_resume_response_data(self, state: dict[str, Any]) -> dict[str, Any]:
        basics = state["basics"]
        target = state["target"]
        skills = state["skills"]
        return {
            "grade_level": basics.get("grade_level"),
            "major": basics.get("major"),
            "school": basics.get("school"),
            "target_role": target.get("role"),
            "job_search_stage": target.get("job_search_stage"),
            "skill_focus": skills.get("hard", []),
            "common_concerns": state.get("unresolved_questions", []),
            "background_summary": state.get("last_update_summary"),
            "resume_state": state,
        }

    def _extract_basics(self, text: str, state: dict[str, Any], changes: list[str]) -> None:
        basics = state["basics"]
        if match := re.search(r"(?:我叫|姓名是|名字叫)\s*([\u4e00-\u9fffA-Za-z]{2,12})", text):
            basics["name"] = match.group(1)
            changes.append("更新姓名")
        for grade in ("大一", "大二", "大三", "大四", "研一", "研二", "研三", "应届"):
            if grade in text:
                basics["grade_level"] = grade
                changes.append("更新年级")
                break
        if match := re.search(r"(?:专业是|学的是|我是)([\u4e00-\u9fffA-Za-z]{2,20})(?:专业|方向)?", text):
            candidate = match.group(1).strip("，。 ")
            if candidate and not any(word in candidate for word in ("想", "负责", "做过")):
                basics["major"] = candidate
                changes.append("更新专业")
        if match := re.search(r"([\u4e00-\u9fffA-Za-z]{2,20}(?:大学|学院|学校))", text):
            basics["school"] = match.group(1)
            changes.append("更新学校")

    def _extract_target(self, text: str, state: dict[str, Any], changes: list[str]) -> None:
        target = state["target"]
        for pattern in self.ROLE_PATTERNS:
            match = pattern.search(text)
            if match:
                role = match.group(1).strip("，。！？ ")
                if role and len(role) <= 24:
                    target["role"] = role
                    changes.append("更新求职意向")
                    break
        if match := self.SALARY_PATTERN.search(text):
            target["salary_expectation"] = match.group(1).replace(" ", "")
            changes.append("更新薪资期望")
        if match := self.CITY_PATTERN.search(text):
            target["city"] = match.group(1)
            changes.append("更新城市偏好")
        for industry, config in self.INDUSTRY_KEYWORDS.items():
            if any(marker in text for marker in config["markers"]):
                target["industry"] = industry
                changes.append(f"识别行业：{config['label']}")
                break

    def _extract_skills(self, text: str, state: dict[str, Any], changes: list[str]) -> None:
        hard = state["skills"]["hard"]
        soft = state["skills"]["soft"]
        before = len(hard) + len(soft)
        for skill in self.HARD_SKILLS:
            if skill.lower() in text.lower() or skill in text:
                hard.append(skill)
        for skill in self.SOFT_SKILLS:
            if skill in text:
                soft.append(skill)
        state["skills"]["hard"] = self._unique(hard)
        state["skills"]["soft"] = self._unique(soft)
        if len(state["skills"]["hard"]) + len(state["skills"]["soft"]) > before:
            changes.append("更新技能标签")

    def _extract_experience(self, text: str, state: dict[str, Any], changes: list[str]) -> None:
        markers = ("负责", "做过", "参与", "主导", "项目", "实习", "工作经历", "优化", "提升", "完成")
        if not any(marker in text for marker in markers):
            return

        metrics = [
            {"value": match.group(1), "unit": match.group(2) or ""}
            for match in self.NUMBER_PATTERN.finditer(text)
        ]
        experience_type = "work"
        if "项目" in text:
            experience_type = "project"
        elif "实习" in text:
            experience_type = "internship"

        experience = {
            "id": f"exp-{len(state['experiences']) + 1}",
            "type": experience_type,
            "raw": text,
            "star": {
                "situation": None,
                "task": self._infer_task(text),
                "action": self._infer_action(text),
                "result": self._infer_result(text, metrics),
            },
            "metrics": metrics,
            "bullets": [self._build_preview_bullet(text, metrics)],
            "created_at": datetime.utcnow().isoformat(),
        }
        state["experiences"].insert(0, experience)
        state["previews"].insert(0, {
            "source": text,
            "content": experience["bullets"][0],
            "type": experience_type,
            "created_at": experience["created_at"],
            "needs_confirmation": len(metrics) == 0,
        })
        state["previews"] = state["previews"][:5]
        changes.append("生成经历片段预览")

    def _extract_preferences(self, text: str, state: dict[str, Any], changes: list[str]) -> None:
        pressure_options = {
            "先拆优先级": "priority_first",
            "沟通资源": "resource_coordination",
            "自己扛": "self_drive",
            "容易焦虑": "needs_structure",
        }
        for label, value in pressure_options.items():
            if label in text:
                state["preferences"]["pressure_response"] = value
                state["skills"]["soft"] = self._unique(state["skills"]["soft"] + ["抗压"])
                changes.append("更新抗压偏好")
                break

    async def _run_llm_extractor(self, text: str, state: dict[str, Any]) -> dict[str, Any] | None:
        try:
            from llm.registry import get_llm_adapter

            llm = get_llm_adapter()
            response = await llm.invoke(
                [{"role": "user", "content": self._build_llm_user_prompt(text, state)}],
                self._build_llm_system_prompt(),
            )
            return self._parse_llm_json(response)
        except Exception:
            return None

    def _build_llm_system_prompt(self) -> str:
        return """
你是 OpenCareer 的简历信息抽取器。只返回 JSON，不要输出解释。

目标：
1. 从用户一句话中补全结构化简历状态。
2. 对模糊表达生成追问。
3. 对经历使用 STAR 法则拆解。
4. 根据行业生成关键词和优先量化指标。
5. 生成一条自然、可信、不夸大的简历 bullet。

严格要求：
- 不要编造数字；用户没给数字时，把 needs_confirmation 设为 true。
- 不要把情绪安慰话写进简历 bullet。
- 用户表达很模糊时，重点生成 follow_up_questions。
- 返回合法 JSON，字段缺失用 null 或空数组。
"""

    def _build_llm_user_prompt(self, text: str, state: dict[str, Any]) -> str:
        compact_state = {
            "target": state.get("target"),
            "basics": state.get("basics"),
            "skills": state.get("skills"),
            "stage": state.get("stage"),
            "latest_experience": state.get("experiences", [None])[0] if state.get("experiences") else None,
        }
        return f"""
用户最新输入：
{text}

当前简历状态摘要：
{json.dumps(compact_state, ensure_ascii=False)}

请返回这个 JSON 结构：
{{
  "target": {{
    "role": null,
    "industry": null,
    "city": null,
    "salary_expectation": null
  }},
  "skills": {{
    "hard": [],
    "soft": []
  }},
  "experience": {{
    "should_create_or_update": false,
    "type": "work|internship|project|null",
    "raw_summary": null,
    "star": {{
      "situation": null,
      "task": null,
      "action": null,
      "result": null
    }},
    "metrics": [
      {{"name": "", "value": "", "unit": "", "needs_confirmation": false}}
    ],
    "bullet": null,
    "needs_confirmation": false
  }},
  "industry_insights": {{
    "keywords": [],
    "preferred_metrics": [],
    "notes": []
  }},
  "follow_up_questions": [],
  "conflicts": [
    {{"type": "", "message": ""}}
  ],
  "confidence": 0.0,
  "reason": null
}}
"""

    def _parse_llm_json(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    def _apply_llm_result(self, result: dict[str, Any], state: dict[str, Any], changes: list[str]) -> None:
        confidence = float(result.get("confidence") or 0)
        if confidence < 0.45:
            state["llm_insights"] = {
                "last_applied": False,
                "confidence": confidence,
                "reason": result.get("reason") or "LLM confidence too low",
            }
            return

        target = result.get("target") or {}
        for key in ("role", "industry", "city", "salary_expectation"):
            value = target.get(key)
            if value and not state["target"].get(key):
                if key == "industry":
                    value = self._normalize_industry(value)
                state["target"][key] = value
                changes.append(f"LLM补充{key}")

        skills = result.get("skills") or {}
        if skills.get("hard"):
            before = len(state["skills"]["hard"])
            state["skills"]["hard"] = self._unique(state["skills"]["hard"] + skills["hard"])
            if len(state["skills"]["hard"]) > before:
                changes.append("LLM补充硬技能")
        if skills.get("soft"):
            before = len(state["skills"]["soft"])
            state["skills"]["soft"] = self._unique(state["skills"]["soft"] + skills["soft"])
            if len(state["skills"]["soft"]) > before:
                changes.append("LLM补充软技能")

        experience = result.get("experience") or {}
        if experience.get("should_create_or_update"):
            self._upsert_llm_experience(experience, state, changes)

        insights = result.get("industry_insights") or {}
        if any(insights.get(key) for key in ("keywords", "preferred_metrics", "notes")):
            state["industry_insights"]["keywords"] = self._unique(
                state["industry_insights"].get("keywords", []) + insights.get("keywords", [])
            )
            state["industry_insights"]["preferred_metrics"] = self._unique(
                state["industry_insights"].get("preferred_metrics", []) + insights.get("preferred_metrics", [])
            )
            state["industry_insights"]["notes"] = self._unique(
                state["industry_insights"].get("notes", []) + insights.get("notes", [])
            )[:5]
            changes.append("LLM补充行业化建议")

        llm_conflicts = [
            item for item in result.get("conflicts", [])
            if item.get("type") and item.get("message")
        ]
        if llm_conflicts:
            state["conflicts"] = self._unique_dicts(state.get("conflicts", []) + llm_conflicts)

        state["llm_insights"] = {
            "last_applied": True,
            "confidence": confidence,
            "reason": result.get("reason"),
        }

    def _upsert_llm_experience(self, experience: dict[str, Any], state: dict[str, Any], changes: list[str]) -> None:
        raw_summary = experience.get("raw_summary")
        bullet = experience.get("bullet")
        star = experience.get("star") or {}
        llm_metrics = [
            {
                "name": item.get("name", ""),
                "value": item.get("value", ""),
                "unit": item.get("unit", ""),
                "needs_confirmation": bool(item.get("needs_confirmation")),
            }
            for item in experience.get("metrics", [])
            if item.get("value") or item.get("name")
        ]

        if state["experiences"] and raw_summary and raw_summary in state["experiences"][0].get("raw", ""):
            target_exp = state["experiences"][0]
        else:
            target_exp = {
                "id": f"exp-{len(state['experiences']) + 1}",
                "type": experience.get("type") or "work",
                "raw": raw_summary or "",
                "star": {"situation": None, "task": None, "action": None, "result": None},
                "metrics": [],
                "bullets": [],
                "created_at": datetime.utcnow().isoformat(),
            }
            state["experiences"].insert(0, target_exp)

        for key in ("situation", "task", "action", "result"):
            if star.get(key):
                target_exp["star"][key] = star[key]
        if llm_metrics:
            target_exp["metrics"] = self._unique_dicts(target_exp.get("metrics", []) + llm_metrics)
        if bullet:
            target_exp["bullets"] = self._unique([bullet] + target_exp.get("bullets", []))
            state["previews"].insert(0, {
                "source": target_exp.get("raw") or raw_summary or "",
                "content": bullet,
                "type": target_exp.get("type", "work"),
                "created_at": datetime.utcnow().isoformat(),
                "needs_confirmation": bool(experience.get("needs_confirmation")) or any(
                    metric.get("needs_confirmation") for metric in llm_metrics
                ),
            })
            state["previews"] = state["previews"][:5]
        changes.append("LLM补强STAR经历")

    def _apply_industry_specialization(self, state: dict[str, Any], changes: list[str]) -> None:
        industry = state["target"].get("industry")
        config = self.INDUSTRY_KEYWORDS.get(industry)
        if not config:
            return
        before = len(state["industry_insights"].get("preferred_metrics", []))
        state["industry_insights"]["preferred_metrics"] = self._unique(
            state["industry_insights"].get("preferred_metrics", []) + config["preferred_metrics"]
        )
        if len(state["industry_insights"]["preferred_metrics"]) > before:
            changes.append("补充行业优先指标")

    def _normalize_industry(self, value: str) -> str:
        mapping = {
            "互联网": "internet",
            "制造": "manufacturing",
            "制造业": "manufacturing",
            "应届": "fresh_graduate",
            "应届生": "fresh_graduate",
        }
        return mapping.get(value, value)

    def _plan_questions(self, text: str, state: dict[str, Any], llm_result: dict[str, Any] | None = None) -> None:
        questions: list[str] = []
        basics = state["basics"]
        target = state["target"]

        if llm_result:
            questions.extend([
                question for question in llm_result.get("follow_up_questions", [])
                if question
            ])

        if not target.get("role"):
            questions.append("你这次主要想投什么岗位？")
        if not target.get("salary_expectation"):
            questions.append("薪资预期大概在哪个范围？")
        if state["experiences"]:
            latest = state["experiences"][0]
            if not latest["metrics"]:
                for trigger, question in self.FUZZY_TRIGGERS.items():
                    if trigger in text:
                        questions.append(question)
                        break
                else:
                    questions.append("这段经历最后有没有能量化的结果，比如人数、周期、效率、成本或转化率？")
        elif target.get("role") and basics.get("grade_level"):
            questions.append("你最近最能代表这个方向的一段项目、实习或工作经历是什么？")
        if not state["preferences"].get("pressure_response"):
            questions.append("遇到多个 deadline 撞在一起时，你更像哪种：先拆优先级、沟通资源、自己扛，还是容易焦虑需要别人帮忙理顺？")

        state["unresolved_questions"] = self._unique(questions)[:3]

    def _detect_conflicts(self, state: dict[str, Any]) -> None:
        conflicts: list[dict[str, str]] = []
        grade = state["basics"].get("grade_level")
        has_full_time = any(exp.get("type") == "work" for exp in state["experiences"])
        if grade in {"大一", "大二", "大三", "大四", "应届"} and has_full_time:
            conflicts.append({
                "type": "education_work_overlap",
                "message": "你看起来还是学生身份，但经历里出现了全职工作。它是实习、兼职，还是正式工作？",
            })
        state["conflicts"] = conflicts

    def _update_stage_and_completion(self, state: dict[str, Any]) -> None:
        basics = state["basics"]
        target = state["target"]
        score = 0
        score += 12 if basics.get("name") else 0
        score += 12 if target.get("role") else 0
        score += 10 if target.get("salary_expectation") else 0
        score += 16 if state["experiences"] else 0
        score += 18 if any(exp.get("metrics") for exp in state["experiences"]) else 0
        score += 14 if state["skills"]["hard"] else 0
        score += 10 if state["preferences"].get("pressure_response") else 0
        score += 8 if not state["conflicts"] else 0
        state["completion"] = min(score, 100)

        if not target.get("role") or not target.get("salary_expectation"):
            state["stage"] = "basic_info"
        elif not state["experiences"]:
            state["stage"] = "experience_discovery"
        elif not any(exp.get("metrics") for exp in state["experiences"]):
            state["stage"] = "experience_deepening"
        elif not state["preferences"].get("pressure_response"):
            state["stage"] = "soft_skill_assessment"
        else:
            state["stage"] = "resume_preview"

    def _build_preview_bullet(self, text: str, metrics: list[dict[str, str]]) -> str:
        if "招聘" in text:
            metric_text = self._format_metric(metrics)
            suffix = f"，完成{metric_text}招聘交付" if metric_text else "，覆盖需求沟通、简历筛选、面试协调与 offer 跟进流程"
            return f"负责招聘相关工作{suffix}。"
        if "优化" in text or "提升效率" in text:
            metric_text = self._format_metric(metrics)
            suffix = f"，推动效率提升{metric_text}" if metric_text else "，沉淀流程优化经验，后续需补充量化结果"
            return f"参与流程优化{suffix}。"
        if "项目" in text:
            metric_text = self._format_metric(metrics)
            suffix = f"，产出{metric_text}结果" if metric_text else "，负责需求拆解、执行推进与结果复盘"
            return f"参与项目工作{suffix}。"
        return text if len(text) <= 80 else text[:77] + "..."

    def _format_metric(self, metrics: list[dict[str, str]]) -> str:
        if not metrics:
            return ""
        first = metrics[0]
        return f"{first['value']}{first['unit']}"

    def _infer_task(self, text: str) -> str | None:
        if "招聘" in text:
            return "招聘交付"
        if "优化" in text:
            return "流程优化"
        if "项目" in text:
            return "项目推进"
        return None

    def _infer_action(self, text: str) -> str | None:
        if any(word in text for word in ("负责", "主导")):
            return text
        return None

    def _infer_result(self, text: str, metrics: list[dict[str, str]]) -> str | None:
        if metrics:
            return self._format_metric(metrics)
        if any(word in text for word in ("提升", "完成", "降低", "增长")):
            return text
        return None

    def _build_event(self, session_id: str, state: dict[str, Any], changes: list[str]) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "state": state,
            "changes": changes,
            "stage": state.get("stage"),
            "completion": state.get("completion", 0),
            "next_questions": state.get("unresolved_questions", []),
            "latest_preview": state.get("previews", [None])[0] if state.get("previews") else None,
            "conflicts": state.get("conflicts", []),
        }

    def _unique(self, items: list[Any]) -> list[Any]:
        result = []
        seen = set()
        for item in items:
            marker = str(item)
            if marker not in seen:
                seen.add(marker)
                result.append(item)
        return result

    def _unique_dicts(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        seen = set()
        for item in items:
            marker = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if marker not in seen:
                seen.add(marker)
                result.append(item)
        return result

    def _deep_update(self, base: dict[str, Any], incoming: dict[str, Any]) -> None:
        for key, value in incoming.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
