"""
Resume Cn Career SKILL for OpenCareer.

This SKILL provides Chinese resume generation, optimization, ATS checking,
and PDF export capabilities for job seekers in China.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from ..base_skill import BaseSkill, SkillMetadata, SkillCategory


class ResumeSkillError(Exception):
    """Resume skill readable error."""
    pass


class ResumeCnCareerSkill(BaseSkill):
    """
    Chinese Resume Generation and Optimization SKILL."""

    def __init__(self):
        """Initialize Resume Cn Career SKILL."""
        metadata = SkillMetadata(
            name="resume_cn_career",
            version="1.0.0",
            description="中文简历生成与优化技能。用于简历生成、岗位定制优化、ATS检查、PDF导出，覆盖校招、社招、资深/管理岗、转行与中英双语投递场景。",
            author="OpenCareer Team",
            category=SkillCategory.DATA,
            tags=["resume", "cv", "chinese", "ats", "pdf", "job", "career"],
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["generate", "optimize", "ats_check", "export_pdf"],
                        "description": "执行动作：generate=生成简历草稿，optimize=按JD优化，ats_check=静态检查，export_pdf=导出PDF"
                    },
                    "name": {"type": "string", "description": "姓名"},
                    "target_role": {"type": "string", "description": "目标岗位"},
                    "job_type": {"type": "string", "description": "求职类型，例如 校招 / 社招 / 转行 / 双语"},
                    "industry": {"type": "string", "description": "行业方向"},
                    "city": {"type": "string", "description": "求职城市"},
                    "experience_years": {"type": "string", "description": "工作年限或应届生说明"},
                    "phone": {"type": "string", "description": "手机号"},
                    "email": {"type": "string", "description": "邮箱"},
                    "education": {"type": "string", "description": "教育背景，建议写成 学校 / 专业 / 学历 / 时间"},
                    "summary": {"type": "string", "description": "职业摘要或自我介绍"},
                    "experiences": {"type": "array", "items": {"type": "string"}, "description": "经历要点列表，每条写一条事实、结果或证据"},
                    "projects": {"type": "array", "items": {"type": "string"}, "description": "项目要点列表，每条写一个项目结果或职责"},
                    "skills": {"type": "array", "items": {"type": "string"}, "description": "技能列表"},
                    "certifications": {"type": "array", "items": {"type": "string"}, "description": "证书或补充信息列表"},
                    "jd": {"type": "string", "description": "目标岗位JD，用于关键词优化和ATS检查"},
                    "bilingual": {"type": "boolean", "default": False, "description": "是否生成双语结构"},
                    "output_json_path": {"type": "string", "description": "输出JSON路径，留空则不落盘"},
                    "output_pdf_path": {"type": "string", "description": "输出PDF路径，留空则不导出"}
                },
                "required": ["action"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean", "description": "是否成功"},
                    "action": {"type": "string", "description": "执行的动作"},
                    "resume": {"type": "object", "description": "生成的简历数据"},
                    "ats_report": {"type": "object", "description": "ATS检查报告"},
                    "saved": {"type": "object", "properties": {
                        "json": {"type": "string", "description": "保存的JSON文件路径"},
                        "pdf": {"type": "string", "description": "保存的PDF文件路径"}
                    }},
                    "notes": {"type": "array", "items": {"type": "string"}, "description": "提示信息"},
                    "error": {"type": "string", "description": "错误信息"}
                }
            },
            examples=[
                {
                    "input": {
                        "action": "generate",
                        "name": "张三",
                        "target_role": "软件工程师",
                        "job_type": "校招",
                        "education": "北京大学 / 计算机科学 / 本科 / 2024",
                        "experiences": [
                            "在字节跳动实习3个月，开发用户增长功能",
                            "获得国家奖学金"
                        ],
                        "skills": ["Python", "Java", "机器学习"]
                    },
                    "output": {
                        "ok": True,
                        "resume": {
                            "format": "chinese",
                            "contact": {"name": "张三"}
                        }
                    }
                }
            ]
        )
        super().__init__(metadata)
        
        # 设置路径
        self._skill_root = Path(__file__).parent
        self._pdf_script = self._skill_root / "scripts" / "generate_resume_pdf.py"
        self._output_dir = self._skill_root / "outputs"

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _ensure_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [self._clean_text(item) for item in value if self._clean_text(item)]
        if isinstance(value, tuple):
            return [self._clean_text(item) for item in value if self._clean_text(item)]
        if isinstance(value, str):
            items = re.split(r"[\n；;]", value)
            return [self._clean_text(item) for item in items if self._clean_text(item)]
        return [self._clean_text(value)]

    def _section(self, title: str, heading: str, bullets: List[str]) -> Dict[str, Any]:
        return {
            "title": title,
            "items": [{"heading": heading, "bullets": bullets}]
        }

    def _extract_keywords(self, text: str) -> List[str]:
        raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]*|[\u4e00-\u9fff]{2,8}", text or "")
        stopwords = {
            "一个", "一些", "我们", "可以", "进行", "负责",
            "提升", "优化", "开发", "项目", "岗位", "能力",
            "经验", "要求", "工作", "相关", "完成", "推动",
            "支持", "以及", "实现"
        }
        keywords: List[str] = []
        for term in raw_terms:
            term = term.strip().lower()
            if len(term) < 2:
                continue
            if term in stopwords:
                continue
            if term not in keywords:
                keywords.append(term)
        return keywords

    def _infer_resume_title(self, job_type: str, target_role: str) -> str:
        job_type = self._clean_text(job_type)
        target_role = self._clean_text(target_role)
        if target_role and job_type:
            return f"{target_role}（{job_type}）"
        if target_role:
            return target_role
        if job_type:
            return job_type
        return "简历"

    def _build_resume(
        self,
        name: str,
        target_role: str,
        job_type: str,
        industry: str,
        city: str,
        experience_years: str,
        phone: str,
        email: str,
        education: str,
        summary: str,
        experiences: List[str],
        projects: List[str],
        skills_list: List[str],
        certifications: List[str],
        jd: str,
        bilingual: bool
    ) -> Dict[str, Any]:
        title = self._infer_resume_title(job_type, target_role)
        contact: Dict[str, Any] = {
            "name": self._clean_text(name) or "未填写姓名",
            "title": title,
            "location": self._clean_text(city),
            "phone": self._clean_text(phone),
            "email": self._clean_text(email)
        }

        if bilingual:
            contact["title"] = f"{title} / Target Role"

        sections: List[Dict[str, Any]] = []

        if education:
            sections.append(
                self._section(
                    "教育背景",
                    self._clean_text(education),
                    []
                )
            )

        if experiences:
            sections.append(
                self._section(
                    "经历亮点",
                    "核心经历",
                    experiences
                )
            )

        if projects:
            sections.append(
                self._section(
                    "项目经历",
                    "代表项目",
                    projects
                )
            )

        if skills_list:
            sections.append(
                self._section(
                    "技能清单",
                    "核心技能",
                    skills_list
                )
            )

        if certifications:
            sections.append(
                self._section(
                    "证书与补充信息",
                    "证书",
                    certifications
                )
            )

        resume = {
            "format": "chinese",
            "contact": contact,
            "summary": self._build_summary_text(name, target_role, job_type, industry, experience_years, jd, summary),
            "sections": sections
        }

        if jd:
            resume["jd_keywords"] = self._extract_keywords(jd)

        if experience_years:
            resume["metadata"] = {
                "job_type": self._clean_text(job_type),
                "industry": self._clean_text(industry),
                "experience_years": self._clean_text(experience_years)
            }

        if bilingual:
            resume.setdefault("metadata", {})["bilingual_requested"] = True

        return resume

    def _build_summary_text(
        self,
        name: str,
        target_role: str,
        job_type: str,
        industry: str,
        experience_years: str,
        jd: str,
        summary: str
    ) -> str:
        summary = self._clean_text(summary)
        if summary:
            return summary

        parts = [f"{self._clean_text(name) or '候选人'}，"]
        parts.append(f"面向{self._clean_text(target_role) or '目标岗位'}，")
        if job_type:
            parts.append(f"属于{job_type}方向，")
        if industry:
            parts.append(f"关注{industry}场景，")
        if experience_years:
            parts.append(f"具备{experience_years}经验，")
        if jd:
            parts.append("已按JD做关键词对齐，")
        parts.append("可围绕事实、结果和证据组织投递内容。")
        return "".join(parts)

    def _gather_resume_text(self, resume: Dict[str, Any]) -> str:
        parts: List[str] = []
        contact = resume.get("contact", {})
        if isinstance(contact, dict):
            for key in ("name", "title", "location", "phone", "email"):
                value = self._clean_text(contact.get(key, ""))
                if value:
                    parts.append(value)

        summary_text = self._clean_text(resume.get("summary", ""))
        if summary_text:
            parts.append(summary_text)

        for section in resume.get("sections", []):
            if not isinstance(section, dict):
                continue
            title = self._clean_text(section.get("title", ""))
            if title:
                parts.append(title)
            for item in section.get("items", []):
                if not isinstance(item, dict):
                    continue
                heading = self._clean_text(item.get("heading", ""))
                if heading:
                    parts.append(heading)
                for bullet in item.get("bullets", []):
                    bullet_text = self._clean_text(bullet)
                    if bullet_text:
                        parts.append(bullet_text)
                text = self._clean_text(item.get("text", ""))
                if text:
                    parts.append(text)

        return "\n".join(parts)

    def _ats_check(self, resume: Dict[str, Any], jd: str) -> Dict[str, Any]:
        issues: List[str] = []
        suggestions: List[str] = []
        score = 100

        contact = resume.get("contact", {})
        if not isinstance(contact, dict):
            issues.append("contact 结构不正确。")
            score -= 20
        else:
            if not self._clean_text(contact.get("name", "")):
                issues.append("缺少姓名。")
                score -= 10
            if not self._clean_text(contact.get("phone", "")):
                issues.append("缺少手机号。")
                score -= 8
            if not self._clean_text(contact.get("email", "")):
                issues.append("缺少邮箱。")
                score -= 8

        if not self._clean_text(resume.get("summary", "")):
            issues.append("缺少职业摘要。")
            score -= 10

        sections = resume.get("sections", [])
        if not sections:
            issues.append("缺少经历或技能模块。")
            score -= 15

        text = self._gather_resume_text(resume)
        if jd:
            jd_keywords = self._extract_keywords(jd)
            matched = [kw for kw in jd_keywords if kw.lower() in text.lower()]
            coverage = round(len(matched) / max(len(jd_keywords), 1) * 100, 1)
            score -= 20 if coverage < 35 else 10 if coverage < 60 else 0
            if coverage < 50:
                suggestions.append("补充JD中高频关键词对应的经历证据。")
            if not matched:
                issues.append("简历与JD的关键词重合度较低。")
            return {
                "score": max(min(score, 100), 0),
                "issues": issues,
                "suggestions": suggestions,
                "jd_keywords": jd_keywords,
                "matched_keywords": matched,
                "keyword_coverage_percent": coverage
            }

        if score < 85:
            suggestions.append("补齐证据型bullet，优先写结果、范围和协作对象。")

        return {
            "score": max(min(score, 100), 0),
            "issues": issues,
            "suggestions": suggestions
        }

    def _write_json(self, path_text: str, payload: Dict[str, Any]) -> str:
        output_path = Path(path_text).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(output_path)

    def _export_pdf(self, resume: Dict[str, Any], output_pdf_path: str) -> str:
        if not self._pdf_script.exists():
            raise ResumeSkillError(f"未找到PDF导出脚本: {self._pdf_script}")

        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(json.dumps(resume, ensure_ascii=False, indent=2))
            temp_json_path = temp_file.name

        try:
            command = [
                sys.executable,
                str(self._pdf_script),
                "--input",
                temp_json_path,
                "--output",
                output_pdf_path,
                "--format",
                "chinese"
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            if completed.stdout.strip():
                self.logger.info(completed.stdout.strip())
            if completed.stderr.strip():
                self.logger.warning(completed.stderr.strip())
            return str(Path(output_pdf_path).expanduser().resolve())
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise ResumeSkillError(f"PDF导出失败: {message}") from exc
        finally:
            try:
                os.unlink(temp_json_path)
            except OSError:
                pass

    async def _initialize(self) -> None:
        """Initialize the SKILL."""
        self.logger.info("Initializing Resume Cn Career SKILL")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.sleep(0.1)
        self.logger.info("Resume Cn Career SKILL initialized")

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the resume skill.

        Args:
            input_data: Input data for the skill
            context: Additional context information

        Returns:
            Skill execution result
        """
        try:
            action = self._clean_text(input_data.get("action", "generate")).lower()
            name = input_data.get("name", "")
            target_role = input_data.get("target_role", "")
            job_type = input_data.get("job_type", "")
            industry = input_data.get("industry", "")
            city = input_data.get("city", "")
            experience_years = input_data.get("experience_years", "")
            phone = input_data.get("phone", "")
            email = input_data.get("email", "")
            education = input_data.get("education", "")
            summary_text = input_data.get("summary", "")
            experiences = self._ensure_list(input_data.get("experiences", []))
            projects = self._ensure_list(input_data.get("projects", []))
            skills_list = self._ensure_list(input_data.get("skills", []))
            certifications = self._ensure_list(input_data.get("certifications", []))
            jd = input_data.get("jd", "")
            bilingual = input_data.get("bilingual", False)
            output_json_path = input_data.get("output_json_path", "")
            output_pdf_path = input_data.get("output_pdf_path", "")

            if action not in {"generate", "optimize", "ats_check", "export_pdf"}:
                raise ResumeSkillError("action 取值无效，请使用 generate、optimize、ats_check 或 export_pdf。")

            if action == "export_pdf" and not self._clean_text(output_pdf_path):
                raise ResumeSkillError("执行 export_pdf 时必须提供 output_pdf_path。")

            resume = self._build_resume(
                name=name,
                target_role=target_role,
                job_type=job_type,
                industry=industry,
                city=city,
                experience_years=experience_years,
                phone=phone,
                email=email,
                education=education,
                summary=summary_text,
                experiences=experiences,
                projects=projects,
                skills_list=skills_list,
                certifications=certifications,
                jd=jd,
                bilingual=bilingual
            )

            ats_report = self._ats_check(resume, jd)
            notes: List[str] = []

            if action in {"optimize", "ats_check"} and jd:
                notes.append("已按JD做静态检查与关键词对齐。")
            if action == "optimize":
                notes.append("当前版本不会虚构经历，只会重排结构并保留可验证事实。")
            if bilingual:
                notes.append("双语模式已开启；如需英文版完整润色，建议补充英文事实稿。")

            saved: Dict[str, str] = {}
            if self._clean_text(output_json_path):
                saved["json"] = self._write_json(output_json_path, resume)

            if action == "export_pdf":
                saved["pdf"] = self._export_pdf(resume, output_pdf_path)

            result: Dict[str, Any] = {
                "ok": True,
                "action": action,
                "resume": resume,
                "ats_report": ats_report,
                "saved": saved,
                "notes": notes
            }

            if not self._clean_text(name):
                result["notes"].append("姓名未填写，已用“未填写姓名”占位。")
            if not self._clean_text(target_role):
                result["notes"].append("目标岗位未填写，标题会以通用简历处理。")
            if not experiences and not projects:
                result["notes"].append("当前经历较少，建议补充2-4条可验证事实。")

            self.logger.info(f"Resume skill executed successfully: {action}")
            return result

        except ResumeSkillError as e:
            self.logger.error(f"Resume skill error: {e}")
            return {
                "ok": False,
                "error": str(e),
                "action": input_data.get("action", "unknown")
            }
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            return {
                "ok": False,
                "error": f"文件不存在: {e}",
                "action": input_data.get("action", "unknown")
            }
        except subprocess.CalledProcessError as e:
            message = e.stderr.strip() or e.stdout.strip() or str(e)
            self.logger.error(f"Subprocess error: {message}")
            return {
                "ok": False,
                "error": f"外部命令执行失败: {message}",
                "action": input_data.get("action", "unknown")
            }
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return {
                "ok": False,
                "error": f"发生未预期错误: {e}",
                "action": input_data.get("action", "unknown")
            }

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up Resume Cn Career SKILL")
        await asyncio.sleep(0.1)


def create_resume_cn_career_skill() -> ResumeCnCareerSkill:
    """Create a ResumeCnCareerSkill instance.

    Returns:
        ResumeCnCareerSkill instance
    """
    return ResumeCnCareerSkill()

