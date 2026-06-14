import asyncio

from db import crud
from services import career_tracking_service as tracking


def test_skill_evidence_chain_merges_follow_up_answers(monkeypatch, tmp_path):
    monkeypatch.setattr(crud, "DB_PATH", str(tmp_path / "app.db"))
    crud.create_session("skill-chain-test", "test-user", "Java 后端工程师")

    async def first_extract(message, resume_update, pending):
        return {
            "evidence": [{
                "skill_name": "Redis",
                "context_key": "订单查询优化",
                "scenario": "订单查询在高峰期响应较慢",
                "action": "使用 Redis 增加查询缓存",
                "confidence": 0.9,
            }],
            "follow_up_question": "这个缓存改动上线后，接口表现发生了什么变化？",
        }

    monkeypatch.setattr(tracking, "_run_skill_extractor", first_extract)
    first = asyncio.run(tracking.update_career_tracking(
        "skill-chain-test",
        "我在订单项目里用 Redis 做了缓存",
        {"state": {}},
    ))
    assert first["skills_updated"] == 1
    assert first["follow_up"]["missing_field"] == "result"

    async def result_extract(message, resume_update, pending):
        return {
            "evidence": [{
                "id": pending["evidence_id"],
                "skill_name": pending["skill_name"],
                "result": "接口响应时间明显下降",
                "metric": "从 800ms 降到 220ms",
                "confidence": 0.95,
            }],
            "follow_up_question": "这部分主要是你独立完成的，还是和团队一起推进的？",
        }

    monkeypatch.setattr(tracking, "_run_skill_extractor", result_extract)
    second = asyncio.run(tracking.update_career_tracking(
        "skill-chain-test",
        "从 800ms 降到了 220ms",
        {"state": {}},
    ))
    assert second["follow_up"]["missing_field"] == "user_role"

    async def role_extract(message, resume_update, pending):
        return {
            "evidence": [{
                "id": pending["evidence_id"],
                "skill_name": pending["skill_name"],
                "user_role": "独立负责",
                "confidence": 0.95,
            }]
        }

    monkeypatch.setattr(tracking, "_run_skill_extractor", role_extract)
    third = asyncio.run(tracking.update_career_tracking(
        "skill-chain-test",
        "这部分是我独立负责的",
        {"state": {}},
    ))
    assert third["follow_up"] is None

    chains = tracking.get_skill_evidence_chains("skill-chain-test")
    assert len(chains["Redis"]) == 1
    assert chains["Redis"][0]["level"] == "strong"
    assert chains["Redis"][0]["completeness"] >= 90

    profile = crud.list_skill_evidence("skill-chain-test")[0]
    assert profile["status"] == "proven"
    assert "220ms" in profile["evidence"]


def test_emotion_mode_disables_skill_follow_up(monkeypatch, tmp_path):
    monkeypatch.setattr(crud, "DB_PATH", str(tmp_path / "app.db"))
    crud.create_session("skill-emotion-test", "test-user")

    async def extractor(message, resume_update, pending):
        return {"evidence": [{"skill_name": "Python", "confidence": 0.8}]}

    monkeypatch.setattr(tracking, "_run_skill_extractor", extractor)
    result = asyncio.run(tracking.update_career_tracking(
        "skill-emotion-test",
        "我会 Python，但现在情绪很差",
        {"state": {}},
        allow_follow_up=False,
    ))
    assert result["skills_updated"] == 1
    assert result["follow_up"] is None
    assert crud.get_pending_skill_follow_up("skill-emotion-test") is None


def test_target_role_statement_is_not_skill_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(crud, "DB_PATH", str(tmp_path / "app.db"))
    crud.create_session("target-intent-test", "test-user")

    async def misleading_extractor(message, resume_update, pending):
        return {"evidence": [{"skill_name": "Java", "confidence": 1.0}]}

    monkeypatch.setattr(tracking, "_run_skill_extractor", misleading_extractor)
    result = asyncio.run(tracking.update_career_tracking(
        "target-intent-test",
        "我想做 Java 后端",
        {"state": {"skills": {"hard": ["Java"], "soft": []}}},
    ))

    assert result == {"jobs_updated": 0, "skills_updated": 0, "evidence_updated": 0, "follow_up": None}
    assert crud.list_skill_evidence_items("target-intent-test") == []
    assert crud.get_pending_skill_follow_up("target-intent-test") is None


def test_empty_legacy_target_evidence_is_pruned(monkeypatch, tmp_path):
    monkeypatch.setattr(crud, "DB_PATH", str(tmp_path / "app.db"))
    crud.create_session("legacy-target-test", "test-user")
    item = crud.save_skill_evidence_item("legacy-target-test", {
        "skill_name": "Java",
        "raw_text": "我想做java后端",
        "confidence": 1.0,
    })
    crud.upsert_skill_evidence("legacy-target-test", {
        "skill_name": "Java",
        "status": "mentioned",
        "evidence": "我想做java后端",
        "source": "evidence_chain",
    })
    crud.save_skill_follow_up("legacy-target-test", {
        "evidence_id": item["id"],
        "skill_name": "Java",
        "missing_field": "result",
        "question": "旧版待清理追问？",
    })

    assert tracking.get_skill_evidence_chains("legacy-target-test") == {}
    assert crud.list_skill_evidence("legacy-target-test") == []
    assert crud.get_pending_skill_follow_up("legacy-target-test") is None


def test_follow_up_starts_with_scenario_for_bare_skill_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(crud, "DB_PATH", str(tmp_path / "app.db"))
    crud.create_session("skill-order-test", "test-user")

    async def extractor(message, resume_update, pending):
        return {
            "evidence": [{"skill_name": "Java", "confidence": 0.8}],
            "follow_up_question": "你在哪段实际经历里使用过 Java？",
        }

    monkeypatch.setattr(tracking, "_run_skill_extractor", extractor)
    result = asyncio.run(tracking.update_career_tracking(
        "skill-order-test",
        "我会 Java",
        {"state": {}},
    ))
    assert result["follow_up"]["missing_field"] == "scenario"


def test_missing_llm_question_does_not_create_canned_follow_up(monkeypatch, tmp_path):
    monkeypatch.setattr(crud, "DB_PATH", str(tmp_path / "app.db"))
    crud.create_session("no-canned-question-test", "test-user")

    async def extractor(message, resume_update, pending):
        return {"evidence": [{"skill_name": "Java", "confidence": 0.8}]}

    monkeypatch.setattr(tracking, "_run_skill_extractor", extractor)
    result = asyncio.run(tracking.update_career_tracking(
        "no-canned-question-test",
        "我在项目里用过 Java",
        {"state": {}},
    ))

    assert result["skills_updated"] == 1
    assert result["follow_up"] is None
    assert crud.get_pending_skill_follow_up("no-canned-question-test") is None
