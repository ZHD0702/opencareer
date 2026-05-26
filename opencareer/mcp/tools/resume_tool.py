"""简历生成与优化 MCP Tool。封装中文简历生成、优化、ATS 检查与 PDF 导出的工作流。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "resume"
PDF_SCRIPT = Path(__file__).resolve().parent / "generate_resume_pdf.py"


class ResumeToolError(Exception):
    """简历工具的可读错误。"""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, tuple):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, str):
        items = re.split(r"[\n；;]+", value)
        return [_clean_text(item) for item in items if _clean_text(item)]
    return [_clean_text(value)]


def _section(title: str, heading: str, bullets: list[str]) -> dict[str, Any]:
    return {
        "title": title,
        "items": [
            {
                "heading": heading,
                "bullets": bullets,
            }
        ],
    }


def _extract_keywords(text: str) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]*|[\u4e00-\u9fff]{2,8}", text or "")
    stopwords = {
        "一个",
        "一些",
        "我们",
        "可以",
        "进行",
        "负责",
        "提升",
        "优化",
        "开发",
        "项目",
        "岗位",
        "能力",
        "经验",
        "要求",
        "工作",
        "相关",
        "完成",
        "推动",
        "支持",
        "以及",
        "实现",
    }
    keywords: list[str] = []
    for term in raw_terms:
        term = term.strip().lower()
        if len(term) < 2:
            continue
        if term in stopwords:
            continue
        if term not in keywords:
            keywords.append(term)
    return keywords


def _infer_resume_title(job_type: str, target_role: str) -> str:
    job_type = _clean_text(job_type)
    target_role = _clean_text(target_role)
    if target_role and job_type:
        return f"{target_role}（{job_type}）"
    if target_role:
        return target_role
    if job_type:
        return job_type
    return "简历"


def _build_summary(
    name: str,
    target_role: str,
    job_type: str,
    industry: str,
    experience_years: str,
    jd: str,
    summary: str,
) -> str:
    summary = _clean_text(summary)
    if summary:
        return summary

    parts = [
        f"{_clean_text(name) or '候选人'}，",
        f"面向{_clean_text(target_role) or '目标岗位'}，",
    ]
    if job_type:
        parts.append(f"属于{job_type}方向，")
    if industry:
        parts.append(f"关注{industry}场景，")
    if experience_years:
        parts.append(f"具备{experience_years}经验，")
    if jd:
        parts.append("已按 JD 做关键词对齐，")
    parts.append("可围绕事实、结果和证据组织投递内容。")
    return "".join(parts)


def _build_resume(
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
    experiences: list[str],
    projects: list[str],
    skills: list[str],
    certifications: list[str],
    jd: str,
    bilingual: bool,
) -> dict[str, Any]:
    title = _infer_resume_title(job_type, target_role)
    contact: dict[str, Any] = {
        "name": _clean_text(name) or "未填写姓名",
        "title": title,
        "location": _clean_text(city),
        "phone": _clean_text(phone),
        "email": _clean_text(email),
    }

    if bilingual:
        contact["title"] = f"{title} / Target Role"

    sections: list[dict[str, Any]] = []

    if education:
        sections.append(
            _section(
                "教育背景",
                _clean_text(education),
                [],
            )
        )

    if experiences:
        sections.append(
            _section(
                "经历亮点",
                "核心经历",
                experiences,
            )
        )

    if projects:
        sections.append(
            _section(
                "项目经历",
                "代表项目",
                projects,
            )
        )

    if skills:
        sections.append(
            _section(
                "技能清单",
                "核心技能",
                skills,
            )
        )

    if certifications:
        sections.append(
            _section(
                "证书与补充信息",
                "证书",
                certifications,
            )
        )

    resume = {
        "format": "chinese",
        "contact": contact,
        "summary": _build_summary(name, target_role, job_type, industry, experience_years, jd, summary),
        "sections": sections,
    }

    if jd:
        resume["jd_keywords"] = _extract_keywords(jd)

    if experience_years:
        resume["metadata"] = {
            "job_type": _clean_text(job_type),
            "industry": _clean_text(industry),
            "experience_years": _clean_text(experience_years),
        }

    if bilingual:
        resume.setdefault("metadata", {})["bilingual_requested"] = True

    return resume


def _gather_resume_text(resume: dict[str, Any]) -> str:
    parts: list[str] = []
    contact = resume.get("contact", {})
    if isinstance(contact, dict):
        for key in ("name", "title", "location", "phone", "email"):
            value = _clean_text(contact.get(key, ""))
            if value:
                parts.append(value)

    summary = _clean_text(resume.get("summary", ""))
    if summary:
        parts.append(summary)

    for section in resume.get("sections", []):
        if not isinstance(section, dict):
            continue
        title = _clean_text(section.get("title", ""))
        if title:
            parts.append(title)
        for item in section.get("items", []):
            if not isinstance(item, dict):
                continue
            heading = _clean_text(item.get("heading", ""))
            if heading:
                parts.append(heading)
            for bullet in item.get("bullets", []):
                bullet_text = _clean_text(bullet)
                if bullet_text:
                    parts.append(bullet_text)
            text = _clean_text(item.get("text", ""))
            if text:
                parts.append(text)

    return "\n".join(parts)


def _ats_check(resume: dict[str, Any], jd: str) -> dict[str, Any]:
    issues: list[str] = []
    suggestions: list[str] = []
    score = 100

    contact = resume.get("contact", {})
    if not isinstance(contact, dict):
        issues.append("contact 结构不正确。")
        score -= 20
    else:
        if not _clean_text(contact.get("name", "")):
            issues.append("缺少姓名。")
            score -= 10
        if not _clean_text(contact.get("phone", "")):
            issues.append("缺少手机号。")
            score -= 8
        if not _clean_text(contact.get("email", "")):
            issues.append("缺少邮箱。")
            score -= 8

    if not _clean_text(resume.get("summary", "")):
        issues.append("缺少职业摘要。")
        score -= 10

    sections = resume.get("sections", [])
    if not sections:
        issues.append("缺少经历或技能模块。")
        score -= 15

    text = _gather_resume_text(resume)
    if jd:
        jd_keywords = _extract_keywords(jd)
        matched = [kw for kw in jd_keywords if kw.lower() in text.lower()]
        coverage = round(len(matched) / max(len(jd_keywords), 1) * 100, 1)
        score -= 20 if coverage < 35 else 10 if coverage < 60 else 0
        if coverage < 50:
            suggestions.append("补充 JD 中高频关键词对应的经历证据。")
        if not matched:
            issues.append("简历与 JD 的关键词重合度较低。")
        return {
            "score": max(min(score, 100), 0),
            "issues": issues,
            "suggestions": suggestions,
            "jd_keywords": jd_keywords,
            "matched_keywords": matched,
            "keyword_coverage_percent": coverage,
        }

    if score < 85:
        suggestions.append("补齐证据型 bullet，优先写结果、范围和协作对象。")

    return {
        "score": max(min(score, 100), 0),
        "issues": issues,
        "suggestions": suggestions,
    }


def _write_json(path_text: str, payload: dict[str, Any]) -> str:
    output_path = Path(path_text).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def _export_pdf(resume: dict[str, Any], output_pdf_path: str) -> str:
    if not PDF_SCRIPT.exists():
        raise ResumeToolError(f"未找到 PDF 导出脚本：{PDF_SCRIPT}")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(json.dumps(resume, ensure_ascii=False, indent=2))
        temp_json_path = temp_file.name

    try:
        command = [
            sys.executable,
            str(PDF_SCRIPT),
            "--input",
            temp_json_path,
            "--output",
            output_pdf_path,
            "--format",
            "chinese",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        if completed.stdout.strip():
            print(completed.stdout.strip())
        if completed.stderr.strip():
            print(completed.stderr.strip(), file=sys.stderr)
        return str(Path(output_pdf_path).expanduser().resolve())
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise ResumeToolError(f"PDF 导出失败：{message}") from exc
    finally:
        try:
            os.remove(temp_json_path)
        except OSError:
            pass


def resume_skill(
    action: Annotated[
        Literal["generate", "optimize", "ats_check", "export_pdf"],
        "执行动作：generate=生成简历草稿，optimize=按 JD 优化，ats_check=静态检查，export_pdf=导出 PDF",
    ] = "generate",
    name: Annotated[str, "姓名"] = "",
    target_role: Annotated[str, "目标岗位"] = "",
    job_type: Annotated[str, "求职类型，例如 校招 / 社招 / 转行 / 双语"] = "",
    industry: Annotated[str, "行业方向"] = "",
    city: Annotated[str, "求职城市"] = "",
    experience_years: Annotated[str, "工作年限或应届生说明"] = "",
    phone: Annotated[str, "手机号"] = "",
    email: Annotated[str, "邮箱"] = "",
    education: Annotated[str, "教育背景，建议写成 学校 / 专业 / 学历 / 时间"] = "",
    summary: Annotated[str, "职业摘要或自我介绍"] = "",
    experiences: Annotated[list[str] | None, "经历要点列表，每条写一条事实、结果或证据"] = None,
    projects: Annotated[list[str] | None, "项目要点列表，每条写一个项目结果或职责"] = None,
    skills: Annotated[list[str] | None, "技能列表"] = None,
    certifications: Annotated[list[str] | None, "证书或补充信息列表"] = None,
    jd: Annotated[str, "目标岗位 JD，用于关键词优化和 ATS 检查"] = "",
    bilingual: Annotated[bool, "是否生成双语结构"] = False,
    output_json_path: Annotated[str, "输出 JSON 路径，留空则自动保存到 outputs/resume/"] = "",
    output_pdf_path: Annotated[str, "输出 PDF 路径，留空则不导出"] = "",
) -> dict[str, Any]:
    """
    封装中文简历生成、优化、ATS 检查与 PDF 导出的工作流。

    Args:
        action: 执行动作。
        name: 姓名。
        target_role: 目标岗位。
        job_type: 求职类型。
        industry: 行业方向。
        city: 求职城市。
        experience_years: 工作年限或应届生说明。
        phone: 手机号。
        email: 邮箱。
        education: 教育背景。
        summary: 职业摘要。
        experiences: 经历要点列表。
        projects: 项目要点列表。
        skills: 技能列表。
        certifications: 证书或补充信息列表。
        jd: 目标岗位 JD。
        bilingual: 是否生成双语结构。
        output_json_path: 输出 JSON 路径。
        output_pdf_path: 输出 PDF 路径。

    Returns:
        包含生成的简历结构、ATS 检查结果和导出状态的字典。
    """
    try:
        action = _clean_text(action).lower()
        experiences_list = _ensure_list(experiences)
        projects_list = _ensure_list(projects)
        skills_list = _ensure_list(skills)
        certifications_list = _ensure_list(certifications)

        if action not in {"generate", "optimize", "ats_check", "export_pdf"}:
            raise ResumeToolError("action 取值无效，请使用 generate、optimize、ats_check 或 export_pdf。")

        resume = _build_resume(
            name=name,
            target_role=target_role,
            job_type=job_type,
            industry=industry,
            city=city,
            experience_years=experience_years,
            phone=phone,
            email=email,
            education=education,
            summary=summary,
            experiences=experiences_list,
            projects=projects_list,
            skills=skills_list,
            certifications=certifications_list,
            jd=jd,
            bilingual=bilingual,
        )

        ats_report = _ats_check(resume, jd)
        notes: list[str] = []

        if action in {"optimize", "ats_check"} and jd:
            notes.append("已按 JD 做静态检查与关键词匹配。")
        if action == "optimize":
            notes.append("当前版本不会虚构经历，只会重排结构并保留可验证事实。")
        if bilingual:
            notes.append("双语模式已开启；如需英文版完整润色，建议补充英文事实稿。")

        saved: dict[str, str] = {}

        contact_name = _clean_text(name) or "候选人"
        role = _clean_text(target_role) or "简历"
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        folder_name = f"{contact_name}-{date_str}"
        output_folder = OUTPUT_DIR / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r'[\\/:*?"<>|]', "_", f"{contact_name}_{role}")
        json_path = str(output_folder / f"{safe_name}.json")
        saved["json"] = _write_json(json_path, resume)

        pdf_path = str(output_folder / f"{safe_name}.pdf")
        if action in {"generate", "optimize", "export_pdf"}:
            saved["pdf"] = _export_pdf(resume, pdf_path)
            notes.append(f"PDF 已生成：{pdf_path}")
            notes.append(f"JSON 已保存：{json_path}")

        result: dict[str, Any] = {
            "ok": True,
            "action": action,
            "resume": resume,
            "ats_report": ats_report,
            "saved": saved,
            "notes": notes,
            "pdf_generated": "pdf" in saved,
            "pdf_path": saved.get("pdf", ""),
            "json_path": saved.get("json", ""),
        }

        if not _clean_text(name):
            result["notes"].append("姓名未填写，已用'未填写姓名'占位。")
        if not _clean_text(target_role):
            result["notes"].append("目标岗位未填写，标题会以通用简历处理。")
        if not experiences_list and not projects_list:
            result["notes"].append("当前经历较少，建议补充 2-4 条可验证事实。")

        return result
    except ResumeToolError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "action": action,
        }
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "error": f"文件不存在：{exc}",
            "action": action,
        }
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        return {
            "ok": False,
            "error": f"外部命令执行失败：{message}",
            "action": action,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"发生未预期错误：{exc}",
            "action": action,
        }
