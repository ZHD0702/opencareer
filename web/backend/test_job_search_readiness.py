from db import crud
from services.job_search_readiness import build_job_match_action, evaluate_job_search_readiness


def seed_ready_profile(session_id: str) -> None:
    crud.create_session(session_id, "test-user", "Java 后端工程师")
    crud.save_resume_state(session_id, {
        "basics": {"grade_level": "应届", "major": "计算机科学", "school": "测试大学"},
        "target": {"role": "Java 后端工程师", "city": "北京", "salary_expectation": "15k-20k"},
        "skills": {"hard": ["Java", "Redis"], "soft": []},
    })
    crud.upsert_skill_evidence(session_id, {
        "skill_name": "Java", "status": "proven", "category": "编程语言",
        "evidence": "独立开发订单接口并将响应时间降低 40%",
    })
    crud.upsert_skill_evidence(session_id, {
        "skill_name": "Redis", "status": "mentioned", "category": "数据库",
        "evidence": "在项目中使用 Redis",
    })


def test_readiness_requires_explicit_search_intent(monkeypatch, tmp_path):
    monkeypatch.setattr(crud, "DB_PATH", str(tmp_path / "app.db"))
    seed_ready_profile("ready-test")
    assessment = evaluate_job_search_readiness("ready-test")
    assert assessment["ready"] is True
    assert assessment["requested"] is False
    assert build_job_match_action("ready-test") is None

    crud.save_message("ready-test", "user", "帮我找一下合适的岗位")
    action = build_job_match_action("ready-test")
    assert action["query_plan"]["role"] == "Java 后端工程师"
    assert action["query_plan"]["city_code"] == "530"


def test_action_is_persisted_and_not_repeated(monkeypatch, tmp_path):
    monkeypatch.setattr(crud, "DB_PATH", str(tmp_path / "app.db"))
    seed_ready_profile("action-test")
    crud.save_message("action-test", "user", "看看有没有适合我的工作")
    action = build_job_match_action("action-test")
    message_id = crud.save_message("action-test", "ai", "资料已经够了，我可以开始帮你找。", actions=[action])

    messages = crud.get_messages("action-test")
    saved = next(message for message in messages if message["id"] == message_id)
    assert saved["actions"][0]["action"] == "start_job_matching"
    assert build_job_match_action("action-test") is None


def test_readiness_reports_missing_profile_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(crud, "DB_PATH", str(tmp_path / "app.db"))
    crud.create_session("missing-test", "test-user")
    crud.save_message("missing-test", "user", "帮我找工作")
    assessment = evaluate_job_search_readiness("missing-test")
    assert assessment["ready"] is False
    assert {"target_role", "city", "background", "skills", "proven_evidence"}.issubset(assessment["missing_fields"])
