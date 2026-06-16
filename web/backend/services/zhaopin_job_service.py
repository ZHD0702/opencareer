from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx


ZHAOPIN_BASE_URL = "https://www.zhaopin.com"
_BASE32_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUV"
_INITIAL_STATE_PATTERN = re.compile(
    r"<script>__INITIAL_STATE__=(.*?)</script>",
    re.DOTALL,
)

MAJOR_EXPANSION = {
    "计算机": [
        "计算机", "软件", "编程", "开发", "算法", "人工智能", "数据分析",
        "后端", "前端", "测试", "运维", "网络", "Java", "Python", "C++",
        "机器学习", "深度学习", "信息系统", "数据库", "IT", "信息管理",
    ],
    "软件": [
        "软件", "编程", "开发", "后端", "前端", "测试", "数据库", "Java",
        "Python", "C++", "Spring", "MySQL", "系统设计",
    ],
    "电子": ["电子", "电路", "嵌入式", "硬件", "芯片", "半导体"],
    "通信": ["通信", "网络", "5G", "光纤", "无线"],
    "自动化": ["自动化", "控制", "PLC", "机器人", "传感器"],
    "机械": ["机械", "结构", "CAD", "制造", "工艺"],
}

CLI_PROFILE_WEIGHTS = {
    "knowledge": 3.0,
    "technology": 2.5,
    "ability": 2.0,
    "industry": 1.5,
    "job_keywords": 2.0,
    "job_position": 1.0,
}

INTERNSHIP_MARKERS = (
    "实习", "实习生", "intern", "internship", "暑期实习", "日常实习",
    "见习", "应届实习", "在校生", "可转正",
)
FULLTIME_MARKERS = (
    "全职", "正式", "社招", "校招", "应届生", "毕业生", "full-time",
    "fulltime", "经验", "年经验", "统招本科", "本科及以上",
)
PART_TIME_MARKERS = ("兼职", "临时", "小时工", "短期")
ANTI_INTERNSHIP_MARKERS = ("不招实习", "非实习", "不接受实习", "实习勿扰")


class ZhaopinSearchError(RuntimeError):
    pass


def encode_zhaopin_keyword(keyword: str) -> str:
    """Mirror the keyword path encoder used by Zhaopin's official search page."""
    bits = "".join(f"{byte:08b}" for byte in keyword.encode("utf-16-be"))
    bits += "0" * ((5 - len(bits) % 5) % 5)
    return "".join(
        _BASE32_ALPHABET[int(bits[index:index + 5], 2)]
        for index in range(0, len(bits), 5)
    )


def build_zhaopin_search_url(city_code: str, keyword: str, page: int = 1) -> str:
    encoded = encode_zhaopin_keyword(keyword.strip())
    return f"{ZHAOPIN_BASE_URL}/sou/jl{city_code}/kw{encoded}/p{max(1, page)}"


def extract_zhaopin_jobs(html: str, source_url: str = "") -> list[dict[str, Any]]:
    match = _INITIAL_STATE_PATTERN.search(html)
    if not match:
        raise ZhaopinSearchError("智联搜索页没有返回可解析的职位数据，可能触发了访问验证")
    try:
        state = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ZhaopinSearchError("智联职位数据解析失败") from exc

    raw_jobs = state.get("positionList") or []
    return [_normalize_job(item, source_url) for item in raw_jobs if item.get("name")]


def _normalize_job(item: dict[str, Any], source_url: str) -> dict[str, Any]:
    detail = item.get("jobDetailData") or {}
    position = detail.get("position") or {}
    description = ((position.get("desc") or {}).get("description") or item.get("jobSummary") or "").strip()
    skills = _unique_strings(
        [entry.get("name") for entry in item.get("jobSkillTags") or []]
        + [entry.get("value") for entry in item.get("skillLabel") or []]
        + ((position.get("desc") or {}).get("labels") or [])
    )
    url = item.get("positionUrl") or item.get("positionURL") or item.get("redirectUrl") or ""
    return {
        "id": str(item.get("jobId") or item.get("number") or item.get("uuid") or url),
        "title": item.get("name") or "",
        "company": item.get("companyName") or "",
        "salary": item.get("salary60") or item.get("salaryReal") or "薪资面议",
        "education": item.get("education") or "",
        "experience": item.get("workingExp") or "",
        "city": item.get("workCity") or "",
        "district": item.get("cityDistrict") or "",
        "industry": item.get("industryName") or "",
        "company_size": item.get("companySize") or "",
        "financing_stage": item.get("financingStage") or "",
        "work_type": item.get("workType") or item.get("propertyName") or "",
        "internship_months": item.get("internshipMonths") or 0,
        "weekly_internship_days": item.get("weeklyInternshipDays") or 0,
        "skills": skills,
        "description": description,
        "published_at": item.get("publishTime") or item.get("firstPublishTime") or "",
        "url": url.replace("http://", "https://", 1),
        "source_url": source_url,
    }


def _unique_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def build_search_queries(plan: dict[str, Any]) -> list[str]:
    role = str(plan.get("role") or "").strip()
    employment_type = str(plan.get("employment_type") or "").strip()
    skills = [str(item).strip() for item in plan.get("skills") or [] if str(item).strip()]

    queries = []
    primary_parts = [role] if role else []
    if employment_type and employment_type not in role:
        primary_parts.append(employment_type)
    primary = " ".join(primary_parts)
    queries.append(primary or role)

    language = next(
        (item for item in [*skills, "Java", "Python", "Go", "C++", "JavaScript"] if item.lower() in role.lower()),
        "",
    )
    suffix = employment_type or ""
    if language:
        queries.append(" ".join(part for part in (language, "开发", suffix) if part))
    if "后端" in role:
        queries.append(" ".join(part for part in (language, "后端开发", suffix) if part))

    supplemental_skills = [skill for skill in skills if skill.lower() not in role.lower()]
    for skill in supplemental_skills[:2]:
        queries.append(" ".join(part for part in (role, skill, employment_type) if part))
    return _unique_strings(queries)[:3]


def _contains_any(text: str, markers: tuple[str, ...] | list[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers if marker)


def _infer_expected_employment_type(plan: dict[str, Any]) -> str:
    explicit = str(plan.get("employment_type") or "").strip().lower()
    role = str(plan.get("role") or "").strip().lower()
    text = " ".join([explicit, role])
    if _contains_any(text, INTERNSHIP_MARKERS):
        return "internship"
    if _contains_any(text, FULLTIME_MARKERS):
        return "fulltime"
    if _contains_any(text, PART_TIME_MARKERS):
        return "parttime"
    return ""


def _infer_job_employment_type(job: dict[str, Any], searchable: str) -> str:
    text = " ".join([
        str(job.get("title") or ""),
        str(job.get("work_type") or ""),
        str(job.get("experience") or ""),
        str(job.get("education") or ""),
        searchable,
    ])
    if _contains_any(text, ANTI_INTERNSHIP_MARKERS):
        return "fulltime"
    if (
        bool(job.get("internship_months") or job.get("weekly_internship_days"))
        or _contains_any(text, INTERNSHIP_MARKERS)
    ):
        return "internship"
    if _contains_any(text, PART_TIME_MARKERS):
        return "parttime"
    if _contains_any(text, FULLTIME_MARKERS):
        return "fulltime"
    return "unknown"


def _score_employment_fit(job: dict[str, Any], plan: dict[str, Any], searchable: str) -> tuple[int, str | None, str | None]:
    expected = _infer_expected_employment_type(plan)
    actual = _infer_job_employment_type(job, searchable)
    if not expected:
        return 0, None, None

    labels = {
        "internship": "实习",
        "fulltime": "全职",
        "parttime": "兼职",
        "unknown": "未明确",
    }
    expected_label = labels.get(expected, expected)
    actual_label = labels.get(actual, actual)

    if expected == actual:
        bonus = 24 if expected == "internship" else 18
        return bonus, f"岗位类型匹配：{expected_label}", None

    if actual == "unknown":
        penalty = -14 if expected == "internship" else -6
        return penalty, None, f"岗位类型未明确标注为{expected_label}"

    penalty = -35 if expected == "internship" else -22
    return penalty, None, f"岗位类型可能不匹配：你要{expected_label}，岗位更像{actual_label}"


def _expanded_profile_keywords(plan: dict[str, Any]) -> list[str]:
    role = str(plan.get("role") or "").strip()
    major = str(plan.get("major") or "").strip()
    industry = str(plan.get("industry") or "").strip()
    skills = [str(item).strip() for item in plan.get("skills") or [] if str(item).strip()]

    keywords = [role, industry, *skills, *_role_tokens(role)]
    for major_key, words in MAJOR_EXPANSION.items():
        if major_key.lower() in major.lower() or major_key.lower() in role.lower():
            keywords.extend(words)
    if major:
        keywords.append(major)
    return _unique_strings(keywords)


def _score_cli_profile_fit(job: dict[str, Any], plan: dict[str, Any], searchable: str) -> tuple[float, list[str]]:
    keywords = _expanded_profile_keywords(plan)
    if not keywords:
        return 0.0, []

    job_skills = [str(item).strip() for item in [*(job.get("skills") or []), *(job.get("skill_tags") or [])] if str(item).strip()]
    fields = {
        "knowledge": " ".join([str(job.get("education") or ""), str(job.get("description") or "")]),
        "technology": " ".join([str(job.get("title") or ""), " ".join(job_skills), str(job.get("description") or "")]),
        "ability": str(job.get("description") or ""),
        "industry": " ".join([str(job.get("industry") or ""), str(job.get("company") or "")]),
        "job_keywords": searchable,
        "job_position": str(job.get("title") or ""),
    }
    raw_score = 0.0
    matched: list[str] = []
    for field, weight in CLI_PROFILE_WEIGHTS.items():
        field_text = fields[field].lower()
        for keyword in keywords:
            if keyword and keyword.lower() in field_text:
                raw_score += weight
                matched.append(keyword)

    # CLI 原始分没有上限；GUI 中将它归一化为 0-30 的补充分。
    normalized = min(30.0, raw_score * 1.6)
    return normalized, _unique_strings(matched)[:6]


@dataclass
class ZhaopinJobService:
    timeout: float = 30.0

    async def search_and_match(self, plan: dict[str, Any], limit: int = 5) -> dict[str, Any]:
        queries = build_search_queries(plan)
        if not queries or not queries[0]:
            raise ZhaopinSearchError("缺少目标岗位，无法搜索")

        city_code = str(plan.get("city_code") or "489")
        jobs_by_id: dict[str, dict[str, Any]] = {}
        searched_urls: list[str] = []
        errors: list[str] = []
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "Referer": f"{ZHAOPIN_BASE_URL}/",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        async with httpx.AsyncClient(headers=headers, timeout=self.timeout, follow_redirects=True) as client:
            for query in queries:
                url = build_zhaopin_search_url(city_code, query)
                searched_urls.append(url)
                try:
                    html, final_url = await self._fetch_html(client, url, headers)
                    for job in extract_zhaopin_jobs(html, final_url):
                        jobs_by_id.setdefault(job["id"], job)
                except (httpx.HTTPError, ZhaopinSearchError) as exc:
                    errors.append(f"{query}: {exc}")

        if not jobs_by_id:
            detail = errors[0] if errors else "没有返回职位"
            raise ZhaopinSearchError(f"智联职位搜索失败：{detail}")

        ranked = [
            score_job(job, plan)
            for job in jobs_by_id.values()
            if job.get("title") and job.get("company") and job.get("url")
        ]
        ranked.sort(key=lambda item: (item["match_score"], item.get("published_at") or ""), reverse=True)
        qualified = [item for item in ranked if item["match_score"] >= 50]
        selected = ranked[:max(1, min(limit, 20))]
        return {
            "source": "智联招聘",
            "matched_at": datetime.now(timezone.utc).isoformat(),
            "queries": queries,
            "searched_urls": searched_urls,
            "total_candidates": len(ranked),
            "qualified_candidates": len(qualified),
            "matches": selected,
            "warnings": errors,
        }

    async def _fetch_html(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
    ) -> tuple[str, str]:
        curl_name = "curl.exe" if os.name == "nt" else "curl"
        curl_path = shutil.which(curl_name)
        curl_error: Exception | None = None
        if os.name == "nt" and curl_path:
            try:
                curl_html, curl_url = await asyncio.to_thread(
                    self._fetch_with_curl,
                    curl_path,
                    url,
                    headers,
                )
                if "__INITIAL_STATE__=" in curl_html:
                    return curl_html, curl_url
                curl_error = ZhaopinSearchError("curl 返回了安全验证页")
            except ZhaopinSearchError as exc:
                curl_error = exc

        try:
            response = await client.get(url)
            response.raise_for_status()
            if "__INITIAL_STATE__=" in response.text:
                return response.text, str(response.url)
            if curl_path and curl_error is None:
                return await asyncio.to_thread(self._fetch_with_curl, curl_path, url, headers)
            if curl_error:
                raise ZhaopinSearchError(
                    f"智联触发访问验证：{curl_error}"
                )
            return response.text, str(response.url)
        except httpx.HTTPError as exc:
            if curl_error:
                raise ZhaopinSearchError(
                    f"curl 与 HTTP 客户端均请求失败：{curl_error}; {exc}"
                ) from exc
            raise

    def _fetch_with_curl(
        self,
        curl_path: str,
        url: str,
        headers: dict[str, str],
    ) -> tuple[str, str]:
        command = [
            curl_path,
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "2",
            "--retry-all-errors",
            "--retry-delay",
            "1",
            "--max-time",
            str(max(5, int(self.timeout))),
            "--user-agent",
            headers["User-Agent"],
            "--referer",
            headers["Referer"],
            url,
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=self.timeout + 5,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
            raise ZhaopinSearchError(f"系统 HTTP 请求失败：{detail or exc}") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise ZhaopinSearchError(f"系统 HTTP 请求失败：{exc}") from exc
        return completed.stdout.decode("utf-8", errors="replace"), url


def score_job(job: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    title = str(job.get("title") or "")
    description = str(job.get("description") or "")
    job_skills = [str(item).strip() for item in [*(job.get("skills") or []), *(job.get("skill_tags") or [])] if str(item).strip()]
    searchable = " ".join([title, description, " ".join(job_skills)]).lower()
    role = str(plan.get("role") or "").strip()
    skills = [str(item).strip() for item in plan.get("skills") or [] if str(item).strip()]
    city = str(plan.get("city") or "").strip()
    major = str(plan.get("major") or "").strip()

    score = 0
    reasons: list[str] = []
    concerns: list[str] = []

    role_tokens = _role_tokens(role)
    title_lower = title.lower()
    matched_role_tokens = [token for token in role_tokens if token.lower() in title_lower]
    if role and role.lower() in title_lower:
        score += 35
        reasons.append("岗位名称与求职方向高度一致")
    elif matched_role_tokens:
        role_score = min(32, 10 + len(matched_role_tokens) * 8)
        score += role_score
        reasons.append(f"岗位名称命中方向关键词：{'、'.join(matched_role_tokens)}")
    else:
        concerns.append("岗位名称与目标方向的直接关联较弱")

    matched_skills = [skill for skill in skills if skill.lower() in searchable]
    missing_skills = [skill for skill in skills if skill not in matched_skills]
    if matched_skills:
        score += min(30, len(matched_skills) * 8)
        reasons.append(f"技能匹配：{'、'.join(matched_skills[:4])}")
    if missing_skills:
        concerns.append(f"暂未在职位描述中看到：{'、'.join(missing_skills[:3])}")

    employment_score, employment_reason, employment_concern = _score_employment_fit(job, plan, searchable)
    score += employment_score
    if employment_reason:
        reasons.append(employment_reason)
    if employment_concern:
        concerns.append(employment_concern)

    if city and (city in str(job.get("city") or "") or city in str(job.get("district") or "")):
        score += 8
        reasons.append(f"工作地点符合：{city}")

    education = str(job.get("education") or "")
    if education in {"学历不限", "不限"}:
        score += 5
        reasons.append("学历要求宽松")
    elif education:
        score += 3

    if major and major.lower() in searchable:
        score += 5
        reasons.append("专业背景与职位描述相关")

    target_salary = _parse_monthly_salary_range(str(plan.get("salary") or ""))
    job_salary = _parse_monthly_salary_range(str(job.get("salary") or ""))
    if target_salary and job_salary:
        if job_salary[1] >= target_salary[0] and target_salary[1] >= job_salary[0]:
            score += 7
            reasons.append("薪资范围与预期有交集")
        elif job_salary[1] < target_salary[0]:
            score -= 5
            concerns.append("职位薪资可能低于当前预期")

    published_age = _published_age_days(str(job.get("published_at") or ""))
    if published_age is not None:
        if published_age <= 7:
            score += 8
            reasons.append("职位近期发布")
        elif published_age <= 30:
            score += 5
            reasons.append("职位发布较新")
        elif published_age > 180:
            score -= 12
            concerns.append("发布时间较早，投递前建议确认职位仍有效")

    base_score = max(0, min(100, score))
    cli_profile_score, cli_profile_matches = _score_cli_profile_fit(job, plan, searchable)
    final_score = round(base_score * 0.7 + cli_profile_score)
    if cli_profile_matches:
        reasons.append(f"画像标签匹配：{'、'.join(cli_profile_matches[:4])}")

    result = dict(job)
    result["match_score"] = max(0, min(100, final_score))
    result["base_rule_score"] = base_score
    result["cli_profile_score"] = round(cli_profile_score, 1)
    result["employment_fit"] = {
        "expected": _infer_expected_employment_type(plan) or "unknown",
        "actual": _infer_job_employment_type(job, searchable),
        "score_delta": employment_score,
    }
    result["match_level"] = "高匹配" if final_score >= 75 else "较匹配" if final_score >= 55 else "可关注"
    result["match_reasons"] = reasons[:5]
    result["concerns"] = concerns[:3]
    result["matched_skills"] = matched_skills
    result["missing_profile_skills"] = missing_skills
    result["matched_profile_keywords"] = cli_profile_matches
    result.pop("description", None)
    return result

def _role_tokens(role: str) -> list[str]:
    ignored = {"工程师", "开发", "实习", "实习生", "岗位", "职位"}
    tokens = re.findall(r"[A-Za-z+#.]+|[\u4e00-\u9fff]{2,}", role)
    expanded = list(tokens)
    for marker in ("前端", "后端", "测试", "算法", "数据", "产品", "运营", "设计", "Java", "Python", "Go"):
        if marker.lower() in role.lower():
            expanded.append(marker)
    return [item for item in _unique_strings(expanded) if item not in ignored]


def _parse_monthly_salary_range(value: str) -> tuple[float, float] | None:
    text = value.lower().replace(" ", "")
    if not text or any(marker in text for marker in ("元/天", "元/时", "元/次", "面议")):
        return None
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)[:2]]
    if not numbers:
        return None
    multiplier = 10000 if "万" in text else 1000 if "k" in text or "千" in text else 1
    values = [number * multiplier for number in numbers]
    if len(values) == 1:
        values.append(values[0])
    return min(values), max(values)


def _published_age_days(value: str) -> int | None:
    if not value:
        return None
    normalized = value.strip().replace("T", " ").replace("Z", "")
    try:
        published = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - published.astimezone(timezone.utc)).days)
