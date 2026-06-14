from api.routes.chat import _build_resume_chat_hint, _strip_internal_tool_markers
from api.routes.sessions import _clean_session_preview


def test_resume_preview_is_not_injected_into_chat_reply():
    hint = _build_resume_chat_hint(
        {
            "changes": ["LLM补强STAR经历"],
            "latest_preview": {"content": "帮我匹配岗位吧，我要找实习"},
            "next_questions": [],
            "conflicts": [],
        }
    )

    assert hint == ""
    assert "简历话术" not in hint


def test_resume_hint_keeps_relevant_follow_up_question():
    hint = _build_resume_chat_hint(
        {
            "changes": ["更新求职意向"],
            "latest_preview": {"content": "不应展示的预览"},
            "next_questions": ["你更倾向在哪个城市找实习？"],
            "conflicts": [],
        }
    )

    assert hint == "你更倾向在哪个城市找实习？"


def test_structured_resume_conflict_is_not_injected_as_chat_copy():
    hint = _build_resume_chat_hint(
        {
            "changes": ["识别经历类型"],
            "next_questions": [],
            "conflicts": [{
                "type": "education_work_overlap",
                "fields": ["grade_level", "experience_type"],
            }],
        }
    )

    assert hint == ""


def test_internal_tool_marker_is_removed_from_assistant_copy():
    text = _strip_internal_tool_markers(
        "[调用工具: resume_skill]\n简历已经生成好了。"
    )

    assert text == "简历已经生成好了。"
    assert "调用工具" not in text


def test_internal_tool_marker_is_removed_from_session_preview():
    preview = _clean_session_preview(
        "[调用工具: resume_skill] 简历已经生成好了。"
    )

    assert preview == "简历已经生成好了。"
