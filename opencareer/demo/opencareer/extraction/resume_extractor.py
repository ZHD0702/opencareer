"""
LLM-powered passive resume information extraction from natural conversation.

Architecture
------------
Delta-based extraction: the LLM receives the user's latest message and the
user's *existing* resume data. It outputs **only what changed** (a delta),
not the full profile. This keeps the prompt focused, the response small,
and the merge logic simple.

For simple (non-repeating) fields::

    {"target_position": "Java后端开发工程师", "target_location": "北京"}

For repeating fields (education_list, work_experience_list, project_experience_list)::

    # Add a new entry
    {"education_list": [{"_op": "add", "school": "北京大学", "major": "软件工程", ...}]}

    # Update an existing entry (match by unique key)
    {"education_list": [{"_op": "update", "_match": {"school": "北京大学"}, "gpa": "3.8/4.0"}]}

Critical design constraints
---------------------------
- **Passive only**: extract what the user VOLUNTEERS. Never output fields
  the user didn't mention. Never ask questions.
- **Delta only**: never output the full profile — only what changed.
- **No fabricated values**: if uncertain, omit the field entirely.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ..memory.resume_schema import (
    FIELDS_BY_MODULE,
    MODULE_LABELS,
    MODULE_ORDER,
    ResumeModule,
    build_flat_profile_for_llm,
    new_empty_resume_data,
)

logger = logging.getLogger("extraction.resume_extractor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DELTA_TAG_ADD = "add"
DELTA_TAG_UPDATE = "update"

# Which top-level keys are always safe to overwrite (non-repeating, single-value)
SIMPLE_OVERWRITE_KEYS = {
    "target_position",
    "target_industry",
    "target_location",
    "name",
    "gender",
    "phone",
    "email",
    "photo",
    "graduation_date",
    "availability",
    "personal_summary",
}

# Dict-type keys that get merged
DICT_MERGE_KEYS = {
    "hard_skills",
    "language_skills",
    "portfolio_links",
}

# List-type keys (non-repeating, non-dict) that get appended
LIST_APPEND_KEYS = {
    "soft_skills",
    "certificates",
    "other_experiences",
}

# Repeating (list-of-dict) keys
REPEATING_FIELD_KEYS = {
    "education_list",
    "work_experience_list",
    "project_experience_list",
}

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

def _build_extraction_system_prompt() -> str:
    """Build the system prompt for the extraction LLM call.

    Dynamically references the full schema from ``resume_schema.py`` so the
    prompt stays in sync as fields are added/modified.
    """
    lines = [
        "# 角色与目标",
        "",
        "你是一个简历信息提取助手。你的任务是从用户的日常对话中**被动提取**简历相关信息。",
        "",
        "## 核心原则（非常重要）",
        "",
        "1. **用户提及才提取** — 只提取用户在本次消息中明确提到的信息。不要猜测，不要编造。",
        "2. **不主动追问** — 不要输出任何问题或引导性内容。如果用户没提，就不输出。",
        "3. **输出增量(delta)** — 只输出本次新增或变更的字段，不要重复输出已有数据。",
        "4. **忠实原文** — 尽量使用用户的原始表述，不要改写或润色。",
        "",
        "## 输出格式",
        "",
        "你必须严格输出 JSON（不要包含 markdown 代码块标记）。输出的 JSON 是一个扁平字典，",
        "key 为字段名，value 为字段值。",
        "",
        "### 非重复字段（普通字段）",
        "",
        "直接输出 key-value 对：",
        '  {"target_position": "Java后端开发工程师", "target_location": "北京"}',
        "",
        "对于字典类型的字段（如 hard_skills、language_skills），输出完整的字典值：",
        '  {"hard_skills": {"Java": "熟练", "Spring Boot": "精通"}}',
        "",
        "对于列表类型的字段（如 soft_skills、certificates），列表中的单个元素为字符串，",
        "输出本次新增的元素（已有元素不需要重复输出）：",
        '  {"soft_skills": ["沟通能力", "团队协作"]}',
        "",
        "### 重复字段（可有多条记录）",
        "",
        "重复字段的值是一个列表，每个元素是一个包含操作标记的字典：",
        "",
        '- **添加新条目**: 使用 `"_op": "add"`，其余字段为子字段值',
        '  {"education_list": [{"_op": "add", "school": "北京大学", "major": "软件工程", ...}]}',
        "",
        '- **更新已有条目**: 使用 `"_op": "update"` + `"_match"`（用于匹配已有记录）+ 要更新的字段',
        '  {"education_list": [{"_op": "update", "_match": {"school": "北京大学"}, "gpa": "3.8/4.0"}]}',
        "",
        "  _match 中的 key-value 用于在已有数据中找到要更新的记录。建议使用 school（教育）、",
        "  company（工作）、project_name（项目）作为匹配键。",
        "",
        "### 不输出空值",
        "",
        "如果用户本次消息中没有提到任何可提取的信息，输出一个空字典 {}。",
        "永远不要输出 null 或空字符串作为字段值。",
        "",
        "## 可提取的字段清单",
        "",
    ]

    # Dynamically describe the schema for the LLM
    for module in MODULE_ORDER:
        label = MODULE_LABELS[module]
        fields = FIELDS_BY_MODULE[module]
        lines.append(f"### {label}")
        for f in fields:
            req = "【必要】" if f.required else "【可选】"
            if f.is_repeating:
                sub_desc = "、".join(
                    f"{sf.key}({sf.label}{'·必要' if sf.required else '·可选'})"
                    for sf in f.sub_fields
                )
                lines.append(f"- {req} {f.key} ({f.label}) — 可多项，子字段: {sub_desc}")
                if f.extraction_hint:
                    lines.append(f"  - 提取提示: {f.extraction_hint}")
            else:
                lines.append(f"- {req} {f.key} ({f.label})")
                if f.extraction_hint:
                    lines.append(f"  - 提取提示: {f.extraction_hint}")
        lines.append("")

    lines.extend([
        "## 注意事项",
        "",
        "- 用户可能用口语化表达提及信息（如「我叫张三」→ name: 张三）",
        "- 时间信息尽量标准化（如「明年六月」→ 2027年6月），但不要过度解读",
        "- 对于工作/项目经历的描述（work_responsibilities、project_content、project_result），",
        "  尽量保留用户的原始表述，不需要转述为STAR格式",
        "- 如果用户提到多条经历，分别作为独立条目添加（每条一个 _op: add）",
        "- 如果用户更新了之前提到的信息，使用 _op: update 更新对应条目",
        "- 如果一条消息中包含多个信息，全部提取",
        "- 如果不确定某个字段的值，宁可不输出也不要猜测",
        "",
        "## 输出要求",
        "",
        '你只输出一个 JSON 对象，不要包含任何其他文字、解释或markdown格式。',
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The extraction prompt is built once at module load time
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = _build_extraction_system_prompt()


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------


def merge_extracted_fields(existing: Dict[str, Any], delta: Dict[str, Any]) -> None:
    """Merge a delta dict into existing resume_data **in-place**.

    Args:
        existing: The current ``resume_data`` dict (modified in-place).
        delta: The delta dict returned by the LLM extraction call.
    """
    if not delta:
        return

    for key, value in delta.items():
        if value is None or value == "" or value == [] or value == {}:
            continue  # skip empty values

        if key in SIMPLE_OVERWRITE_KEYS:
            # Simple scalar field — just overwrite
            existing[key] = value

        elif key in DICT_MERGE_KEYS:
            # Dict field — merge (existing dict gets updated)
            existing_val = existing.get(key, {})
            if not isinstance(existing_val, dict):
                existing_val = {}
            if isinstance(value, dict):
                existing_val.update(value)
                existing[key] = existing_val

        elif key in LIST_APPEND_KEYS:
            # List field — append unique items
            existing_val = existing.get(key, [])
            if not isinstance(existing_val, list):
                existing_val = []
            if isinstance(value, list):
                seen = set(existing_val)
                for item in value:
                    if item not in seen:
                        existing_val.append(item)
                        seen.add(item)
                existing[key] = existing_val

        elif key in REPEATING_FIELD_KEYS:
            # Repeating (list-of-dict) field
            existing_val = existing.get(key, [])
            if not isinstance(existing_val, list):
                existing_val = []

            if isinstance(value, list):
                for entry in value:
                    if not isinstance(entry, dict):
                        continue
                    op = entry.pop("_op", None)
                    match_key = entry.pop("_match", None)

                    if op == DELTA_TAG_ADD:
                        # Add a new entry
                        existing_val.append(entry)

                    elif op == DELTA_TAG_UPDATE and match_key:
                        # Update matching entry
                        _apply_update_to_matching(existing_val, match_key, entry)

                existing[key] = existing_val

        else:
            # Unknown key — still apply as simple overwrite
            existing[key] = value


def _apply_update_to_matching(
    entries: List[Dict[str, Any]],
    match_key: Dict[str, Any],
    updates: Dict[str, Any],
) -> None:
    """Find an entry in *entries* matching *match_key* and apply *updates*.

    Uses the first matching entry (matched by ALL key-value pairs in match_key).
    If no match is found, the updates are silently ignored (the LLM may
    hallucinate a non-existent entry).
    """
    for entry in entries:
        if isinstance(entry, dict) and all(
            entry.get(k) == v for k, v in match_key.items()
        ):
            entry.update(updates)
            return


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


async def extract_resume_fields(
    user_input: str,
    existing_data: Optional[Dict[str, Any]] = None,
    llm_client=None,
) -> Dict[str, Any]:
    """Extract resume-related fields from *user_input* using the LLM.

    This is a passive extraction: the LLM only returns fields the user
    explicitly mentioned in this message. It never asks questions and
    never fabricates information.

    Args:
        user_input: The user's latest message text.
        existing_data: The user's current ``resume_data`` dict (or ``None``
            if first-time extraction). Used to avoid re-extracting known info.
        llm_client: The ``LLMClient`` instance for API calls. Must support
            ``chat_json(system_prompt, user_input, context)``.

    Returns:
        A delta dict containing only newly extracted/changed fields.
        Returns ``{}`` if nothing extractable was found.

    Raises:
        RuntimeError: If *llm_client* is ``None``.
    """
    if llm_client is None:
        raise RuntimeError("llm_client is required for LLM-based extraction")

    # Build context string from existing data so the LLM knows what's already known
    if existing_data:
        context = build_flat_profile_for_llm(existing_data)
    else:
        context = "暂无已知简历信息"

    extract_msg = (
        "请从以下用户消息中提取简历相关信息。\n\n"
        f"已知简历数据：\n{context}\n\n"
        f"用户消息：{user_input}"
    )

    logger.debug("Calling LLM for resume field extraction")

    try:
        result = await llm_client.chat_json(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_input=extract_msg,
            temperature=0.1,  # low temperature for consistent extraction
            max_tokens=2048,
        )

        if not isinstance(result, dict):
            logger.warning(f"LLM returned non-dict result: {type(result)}")
            return {}

        # Filter out empty results
        result = {k: v for k, v in result.items() if v not in (None, "", [], {})}
        if not result:
            logger.debug("LLM extraction returned empty (no new info)")
            return {}

        logger.info(f"LLM extracted {len(result)} field(s): {list(result.keys())}")
        return result

    except Exception as exc:
        logger.error(f"LLM extraction failed: {exc}", exc_info=True)
        return {}
