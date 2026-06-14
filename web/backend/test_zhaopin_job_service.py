import json
import asyncio
from datetime import datetime, timezone

from services.zhaopin_job_service import (
    build_search_queries,
    build_zhaopin_search_url,
    encode_zhaopin_keyword,
    extract_zhaopin_jobs,
    score_job,
    ZhaopinJobService,
    ZhaopinSearchError,
)


def test_keyword_encoder_matches_zhaopin_official_page_algorithm():
    assert encode_zhaopin_keyword("Java后端实习") == "01500O80EO062L0EFBNLN7IEC0"
    assert build_zhaopin_search_url("635", "Java后端实习", 2).endswith(
        "/sou/jl635/kw01500O80EO062L0EFBNLN7IEC0/p2"
    )


def test_extract_and_score_realistic_position_state():
    state = {
        "positionList": [{
            "jobId": 42,
            "name": "Java 后端开发实习生",
            "companyName": "示例科技有限公司",
            "salary60": "150-200元/天",
            "education": "本科",
            "workingExp": "经验不限",
            "workCity": "南京",
            "cityDistrict": "建邺",
            "positionUrl": "http://www.zhaopin.com/jobdetail/42.htm",
            "internshipMonths": 3,
            "weeklyInternshipDays": 4,
            "jobSkillTags": [{"name": "Java"}, {"name": "Spring Boot"}, {"name": "MySQL"}],
            "jobDetailData": {
                "position": {"desc": {"description": "负责 Spring Boot 服务开发，使用 MySQL。", "labels": []}}
            },
        }]
    }
    html = f"<html><script>__INITIAL_STATE__={json.dumps(state, ensure_ascii=False)}</script></html>"
    jobs = extract_zhaopin_jobs(html, "https://www.zhaopin.com/sou/example")
    result = score_job(jobs[0], {
        "role": "Java 后端开发",
        "city": "南京",
        "employment_type": "实习",
        "skills": ["Java", "Spring Boot", "MySQL", "Redis"],
    })

    assert jobs[0]["url"].startswith("https://")
    assert result["match_score"] >= 75
    assert result["match_level"] == "高匹配"
    assert result["matched_skills"] == ["Java", "Spring Boot", "MySQL"]
    assert "Redis" in result["missing_profile_skills"]


def test_search_queries_use_role_employment_type_and_skills_without_duplicates():
    queries = build_search_queries({
        "role": "Java 后端",
        "employment_type": "实习",
        "skills": ["Java", "Spring Boot", "MySQL"],
    })
    assert queries == ["Java 后端 实习", "Java 开发 实习", "Java 后端开发 实习"]


def test_recent_real_job_gets_recency_signal():
    result = score_job({
        "title": "Java 后端实习生",
        "company": "示例公司",
        "salary": "200元/天",
        "education": "本科",
        "experience": "经验不限",
        "city": "南京",
        "district": "建邺",
        "skills": ["Java"],
        "description": "使用 Java 开发后端服务",
        "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "url": "https://www.zhaopin.com/jobdetail/example.htm",
    }, {
        "role": "Java 后端",
        "city": "南京",
        "employment_type": "实习",
        "skills": ["Java"],
    })

    assert "职位近期发布" in result["match_reasons"]


def test_windows_curl_failure_falls_back_to_http_client(monkeypatch):
    html = '<script>__INITIAL_STATE__={"positionList": []}</script>'

    class FakeResponse:
        text = html
        url = "https://www.zhaopin.com/sou/example"

        def raise_for_status(self):
            return None

    class FakeClient:
        async def get(self, url):
            return FakeResponse()

    service = ZhaopinJobService()
    monkeypatch.setattr("services.zhaopin_job_service.os.name", "nt")
    monkeypatch.setattr("services.zhaopin_job_service.shutil.which", lambda _: "curl.exe")
    monkeypatch.setattr(
        service,
        "_fetch_with_curl",
        lambda *args: (_ for _ in ()).throw(ZhaopinSearchError("TLS failure")),
    )

    body, final_url = asyncio.run(service._fetch_html(FakeClient(), "https://example", {}))

    assert body == html
    assert final_url.endswith("/sou/example")


def test_search_returns_five_real_jobs_even_when_some_are_below_qualified_threshold(monkeypatch):
    positions = []
    for index in range(6):
        positions.append({
            "jobId": index + 1,
            "name": f"Java 开发岗位 {index + 1}",
            "companyName": f"真实公司 {index + 1}",
            "salary60": "150-200元/天",
            "education": "本科",
            "workingExp": "经验不限",
            "workCity": "南京",
            "positionUrl": f"https://www.zhaopin.com/jobdetail/{index + 1}.htm",
            "jobSkillTags": [{"name": "Java"}] if index < 2 else [],
            "jobDetailData": {"position": {"desc": {"description": "Java 开发", "labels": []}}},
        })
    html = f"<script>__INITIAL_STATE__={json.dumps({'positionList': positions}, ensure_ascii=False)}</script>"

    service = ZhaopinJobService()

    async def fake_fetch(*args):
        return html, "https://www.zhaopin.com/sou/test"

    monkeypatch.setattr(service, "_fetch_html", fake_fetch)
    result = asyncio.run(service.search_and_match({
        "role": "Java 后端",
        "city": "南京",
        "city_code": "635",
        "employment_type": "实习",
        "skills": ["Java", "Spring Boot"],
    }, limit=5))

    assert len(result["matches"]) == 5
    assert all(job["url"].startswith("https://www.zhaopin.com/jobdetail/") for job in result["matches"])
