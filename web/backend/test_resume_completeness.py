from services.mcp_resume_service import (
    build_incomplete_resume_response,
    build_resume_skill_payload,
    validate_resume_payload,
)


def test_incomplete_resume_is_blocked():
    payload = build_resume_skill_payload(
        {
            "basics": {"name": None, "school": "测试大学", "major": "计算机", "grade_level": "本科"},
            "target": {"role": "Java 后端"},
            "skills": {"hard": ["Java"], "soft": []},
            "experiences": [{"type": "project", "bullets": ["完成后端项目"]}],
        }
    )

    validation = validate_resume_payload(payload)

    assert validation["complete"] is False
    assert validation["missing_labels"] == ["姓名", "手机号", "邮箱"]
    assert "还缺：姓名、手机号、邮箱" in build_incomplete_resume_response(validation)


def test_complete_resume_can_be_exported():
    payload = build_resume_skill_payload(
        {
            "basics": {
                "name": "张三",
                "phone": "13800138000",
                "email": "zhangsan@example.com",
                "school": "测试大学",
                "major": "计算机",
                "grade_level": "本科",
            },
            "target": {"role": "Java 后端"},
            "skills": {"hard": ["Java", "Spring Boot"], "soft": []},
            "experiences": [{"type": "project", "bullets": ["完成后端项目"]}],
        },
        {
            "Java": [{"level": "strong"}],
            "Spring Boot": [{"level": "used"}],
        },
    )

    assert validate_resume_payload(payload)["complete"] is True
    assert payload["skills"] == ["Java（熟练）", "Spring Boot（掌握）"]


def test_skill_without_practice_evidence_is_marked_as_basic_awareness():
    payload = build_resume_skill_payload(
        {
            "basics": {},
            "target": {},
            "skills": {"hard": ["Redis"], "soft": []},
        },
        {},
    )

    assert payload["skills"] == ["Redis（了解）"]
