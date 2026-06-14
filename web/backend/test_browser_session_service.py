import asyncio
import json

import pytest

from services.browser_session_service import BrowserSessionService, BrowserTask


@pytest.mark.asyncio
async def test_browser_error_never_has_an_empty_message(monkeypatch):
    service = BrowserSessionService()
    task = BrowserTask(id="task", owner_session_id="session", query_plan={"role": "Java"})

    def fail_browser(task):
        raise NotImplementedError

    monkeypatch.setattr(service, "_run_search_sync", fail_browser)
    await service._run_search(task)

    assert task.status == "error"
    assert task.error == "NotImplementedError"
    service._executor.shutdown(wait=False, cancel_futures=True)


def test_start_script_disables_windows_reload_mode():
    from pathlib import Path

    script = (Path(__file__).resolve().parents[2] / "start_all.bat").read_text(encoding="utf-8")
    backend_command = next(line for line in script.splitlines() if "uvicorn main:app" in line)
    assert "--reload" not in backend_command
    assert "--port 8002" in backend_command


def test_frontend_proxy_targets_active_backend_port():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    vite_config = (project_root / "web" / "front" / "vite.config.ts").read_text(encoding="utf-8")
    assert "http://127.0.0.1:8002" in vite_config


@pytest.mark.asyncio
async def test_resume_reuses_the_existing_page_and_context(monkeypatch):
    service = BrowserSessionService()

    class FakePage:
        def is_closed(self):
            return False

    page = FakePage()
    context = object()
    task = BrowserTask(id="task", owner_session_id="session", query_plan={"role": "Java"}, status="needs_user")
    task.page = page
    task.context = context
    service._tasks[task.id] = task

    def continue_search(current):
        assert current.page is page
        assert current.context is context
        current.status = "ready"
        current.status_text = "done"

    monkeypatch.setattr(service, "_continue_search_sync", continue_search)
    response = await service.resume(task.id)
    assert response["status"] == "resuming"
    await task.runner

    assert task.status == "ready"
    assert task.page is page
    assert task.context is context
    service._executor.shutdown(wait=False, cancel_futures=True)


def test_browser_page_results_are_ranked_and_limited_to_five():
    service = BrowserSessionService()
    positions = [{
        "jobId": index,
        "name": f"Java 后端实习生 {index}",
        "companyName": f"示例公司 {index}",
        "salary60": "150-200元/天",
        "education": "本科",
        "workingExp": "经验不限",
        "workCity": "南京",
        "positionUrl": f"https://www.zhaopin.com/jobdetail/{index}.htm",
        "internshipMonths": 3,
        "jobSkillTags": [{"name": "Java"}, {"name": "Spring Boot"}],
        "jobDetailData": {
            "position": {"desc": {"description": "使用 Java 和 Spring Boot 开发后端服务", "labels": []}}
        },
    } for index in range(1, 7)]
    html = f"<script>__INITIAL_STATE__={json.dumps({'positionList': positions}, ensure_ascii=False)}</script>"

    class FakePage:
        url = "https://www.zhaopin.com/sou/jl635/example"

        def is_closed(self):
            return False

        def content(self):
            return html

    task = BrowserTask(
        id="task",
        owner_session_id="session",
        query_plan={
            "role": "Java 后端",
            "city": "南京",
            "employment_type": "实习",
            "skills": ["Java", "Spring Boot"],
        },
    )
    task.page = FakePage()

    service._update_matches_from_page_sync(task, limit=5)

    assert task.total_candidates == 6
    assert len(task.matches) == 5
    assert all(job["url"].startswith("https://www.zhaopin.com/jobdetail/") for job in task.matches)
    service._executor.shutdown(wait=False, cancel_futures=True)
