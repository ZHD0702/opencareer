"""
Resume information schema definition for OpenCareer.

Defines the complete data structure for resume-related user information
across 7 modules extracted from 简历制作用户信息采集清单.

This is the **Schema** layer of the Schema + Data 双层结构 architecture.
The Data layer lives in ConversationContext.user_profile["resume_data"]
and is persisted via the resume store.

Architecture:
    Schema (this file)         → field definitions, metadata, validation rules
    Data  (user_profile)       → per-user values stored in ConversationContext
    Persistence (resume_store) → save/load from structured memory
    Extraction (LLM prompt)    → passive extraction from natural conversation
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("memory.resume_schema")


# ============================================================================
# Module definitions
# ============================================================================

class ResumeModule(str, Enum):
    """The 7 modules of the resume information model."""
    JOB_INTENT = "job_intent"          # 求职意向定位
    PERSONAL_INFO = "personal_info"     # 个人基本信息
    EDUCATION = "education"             # 教育背景
    WORK_EXPERIENCE = "work_experience" # 工作/实习经历
    PROJECT_EXPERIENCE = "project_experience"  # 项目经历
    SKILLS_CERTIFICATES = "skills_certificates" # 专业技能与证书
    OTHER_INFO = "other_info"           # 其他个性化信息


MODULE_LABELS: Dict[ResumeModule, str] = {
    ResumeModule.JOB_INTENT: "求职意向定位",
    ResumeModule.PERSONAL_INFO: "个人基本信息",
    ResumeModule.EDUCATION: "教育背景",
    ResumeModule.WORK_EXPERIENCE: "工作/实习经历",
    ResumeModule.PROJECT_EXPERIENCE: "项目经历",
    ResumeModule.SKILLS_CERTIFICATES: "专业技能与证书",
    ResumeModule.OTHER_INFO: "其他个性化信息",
}

MODULE_ORDER: List[ResumeModule] = [
    ResumeModule.JOB_INTENT,
    ResumeModule.PERSONAL_INFO,
    ResumeModule.EDUCATION,
    ResumeModule.WORK_EXPERIENCE,
    ResumeModule.PROJECT_EXPERIENCE,
    ResumeModule.SKILLS_CERTIFICATES,
    ResumeModule.OTHER_INFO,
]


# ============================================================================
# Field type constants
# ============================================================================

class FieldType(str, Enum):
    STRING = "string"
    LIST = "list"           # list of strings
    LIST_DICT = "list_dict" # list of dicts (repeating group, e.g. work experience entries)
    DICT = "dict"           # nested dict


# ============================================================================
# Individual field definition
# ============================================================================

class ResumeFieldDef:
    """Definition of a single resume information field.

    Attributes:
        key: Machine-readable field name (snake_case)
        label: Human-readable Chinese label
        field_type: FieldType enum
        required: True if this field is required for resume generation
        module: Which module this field belongs to
        description: Detailed description of what information to capture
        extraction_hint: Guidance for the LLM extraction prompt
        is_repeating: True if this field can have multiple entries (e.g. work history)
        sub_fields: For repeating fields, the sub-field definitions
    """

    def __init__(
        self,
        key: str,
        label: str,
        field_type: FieldType = FieldType.STRING,
        required: bool = False,
        module: ResumeModule = ResumeModule.OTHER_INFO,
        description: str = "",
        extraction_hint: str = "",
        is_repeating: bool = False,
        sub_fields: Optional[List[ResumeFieldDef]] = None,
    ):
        self.key = key
        self.label = label
        self.field_type = field_type
        self.required = required
        self.module = module
        self.description = description
        self.extraction_hint = extraction_hint
        self.is_repeating = is_repeating
        self.sub_fields = sub_fields or []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for introspection / prompt building."""
        return {
            "key": self.key,
            "label": self.label,
            "field_type": self.field_type.value,
            "required": self.required,
            "module": self.module.value,
            "description": self.description,
            "extraction_hint": self.extraction_hint,
            "is_repeating": self.is_repeating,
            "sub_fields": [sf.to_dict() for sf in self.sub_fields],
        }

    def __repr__(self) -> str:
        return (
            f"ResumeFieldDef(key={self.key}, label={self.label}, "
            f"required={self.required}, module={self.module.value})"
        )


# ============================================================================
# Complete field definitions
# ============================================================================

# --- Module 1: 求职意向定位 ---

JOB_INTENT_FIELDS = [
    ResumeFieldDef(
        key="target_position",
        label="目标岗位名称",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.JOB_INTENT,
        description="用户期望投递的目标岗位名称，如「Java后端开发工程师」「产品经理」",
        extraction_hint="用户明确提到的求职目标岗位，可能出现在「想找XX工作」「目标是XX岗」「投递XX」等表述中",
    ),
    ResumeFieldDef(
        key="target_industry",
        label="目标行业/领域",
        field_type=FieldType.STRING,
        required=False,
        module=ResumeModule.JOB_INTENT,
        description="用户期望的行业方向，如「互联网」「金融科技」「智能制造」",
        extraction_hint="用户提到的行业偏好，如「想去互联网行业」「在金融领域发展」",
    ),
    ResumeFieldDef(
        key="target_location",
        label="目标工作地点",
        field_type=FieldType.STRING,
        required=False,
        module=ResumeModule.JOB_INTENT,
        description="用户期望的工作城市或地区，如「北京」「上海」「杭州」",
        extraction_hint="用户提到的城市偏好，如「想去北京工作」「希望在杭州」",
    ),
]

# --- Module 2: 个人基本信息 ---

PERSONAL_INFO_FIELDS = [
    ResumeFieldDef(
        key="name",
        label="姓名",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.PERSONAL_INFO,
        description="用户姓名",
        extraction_hint="用户直接告知的姓名，通常出现在「我叫XX」「我是XX」的表述中",
    ),
    ResumeFieldDef(
        key="gender",
        label="性别",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.PERSONAL_INFO,
        description="用户性别：男/女",
        extraction_hint="用户可能直接提及或可从上下文中推断",
    ),
    ResumeFieldDef(
        key="phone",
        label="有效联系电话",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.PERSONAL_INFO,
        description="用户联系电话号码",
        extraction_hint="11位手机号码或固定电话号码",
    ),
    ResumeFieldDef(
        key="email",
        label="专业联系邮箱",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.PERSONAL_INFO,
        description="用户用于求职的电子邮箱",
        extraction_hint="包含@符号的邮箱地址，优先使用非QQ邮箱的专业邮箱",
    ),
    ResumeFieldDef(
        key="photo",
        label="证件照/职业照",
        field_type=FieldType.STRING,
        required=False,
        module=ResumeModule.PERSONAL_INFO,
        description="是否已有证件照或职业照（记录路径或状态）",
        extraction_hint="用户提到「有照片」「证件照」等信息时记录",
    ),
    ResumeFieldDef(
        key="graduation_date",
        label="毕业时间",
        field_type=FieldType.STRING,
        required=False,
        module=ResumeModule.PERSONAL_INFO,
        description="预计或已毕业的时间，如「2026年6月」",
        extraction_hint="用户提到的毕业年份和月份，如「2026年毕业」「明年6月毕业」",
    ),
    ResumeFieldDef(
        key="availability",
        label="到岗状态",
        field_type=FieldType.STRING,
        required=False,
        module=ResumeModule.PERSONAL_INFO,
        description="是否可以随时到岗或需要提前多久通知，如「随时到岗」「提前两周」",
        extraction_hint="用户提到的可到岗时间信息",
    ),
]

# --- Module 3: 教育背景 ---

EDUCATION_SUB_FIELDS = [
    ResumeFieldDef(
        key="school",
        label="学校全称",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.EDUCATION,
        description="就读或毕业的学校全称，如「北京大学」「华中科技大学」",
    ),
    ResumeFieldDef(
        key="major",
        label="就读专业",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.EDUCATION,
        description="就读的专业名称，如「软件工程」「计算机科学与技术」",
    ),
    ResumeFieldDef(
        key="degree",
        label="学历学位",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.EDUCATION,
        description="学历学位，如「本科」「硕士」「博士」，或具体学位名称",
    ),
    ResumeFieldDef(
        key="education_period",
        label="就读起止时间",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.EDUCATION,
        description="入学到毕业的时间段，如「2022.09 - 2026.06」",
    ),
    ResumeFieldDef(
        key="gpa",
        label="GPA/专业排名",
        field_type=FieldType.STRING,
        required=False,
        module=ResumeModule.EDUCATION,
        description="GPA成绩或专业排名，如「3.8/4.0」「专业前10%」",
    ),
    ResumeFieldDef(
        key="relevant_courses",
        label="相关主修课程",
        field_type=FieldType.LIST,
        required=False,
        module=ResumeModule.EDUCATION,
        description="与目标岗位相关的主修课程列表",
    ),
    ResumeFieldDef(
        key="honors",
        label="在校荣誉奖励",
        field_type=FieldType.LIST,
        required=False,
        module=ResumeModule.EDUCATION,
        description="获得的奖学金、竞赛奖项、荣誉称号等",
    ),
]

EDUCATION_FIELDS = [
    ResumeFieldDef(
        key="education_list",
        label="教育背景",
        field_type=FieldType.LIST_DICT,
        required=True,
        module=ResumeModule.EDUCATION,
        description="一段或多段教育经历",
        is_repeating=True,
        sub_fields=EDUCATION_SUB_FIELDS,
    ),
]

# --- Module 4: 工作/实习经历 ---

WORK_SUB_FIELDS = [
    ResumeFieldDef(
        key="company",
        label="公司全称",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.WORK_EXPERIENCE,
    ),
    ResumeFieldDef(
        key="department",
        label="所属部门",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.WORK_EXPERIENCE,
    ),
    ResumeFieldDef(
        key="position",
        label="担任职位",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.WORK_EXPERIENCE,
    ),
    ResumeFieldDef(
        key="work_period",
        label="工作/实习起止时间",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.WORK_EXPERIENCE,
    ),
    ResumeFieldDef(
        key="work_responsibilities",
        label="核心工作职责与成果STAR",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.WORK_EXPERIENCE,
        description="按STAR法则描述的工作职责、行动和量化成果",
    ),
    ResumeFieldDef(
        key="company_industry",
        label="公司所属行业/规模",
        field_type=FieldType.STRING,
        required=False,
        module=ResumeModule.WORK_EXPERIENCE,
    ),
]

WORK_EXPERIENCE_FIELDS = [
    ResumeFieldDef(
        key="work_experience_list",
        label="工作/实习经历",
        field_type=FieldType.LIST_DICT,
        required=True,
        module=ResumeModule.WORK_EXPERIENCE,
        description="一段或多段工作/实习经历",
        is_repeating=True,
        sub_fields=WORK_SUB_FIELDS,
    ),
]

# --- Module 5: 项目经历 ---

PROJECT_SUB_FIELDS = [
    ResumeFieldDef(
        key="project_name",
        label="项目全称",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.PROJECT_EXPERIENCE,
    ),
    ResumeFieldDef(
        key="project_role",
        label="担任角色",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.PROJECT_EXPERIENCE,
        description="在项目中担任的角色，如「项目负责人」「核心开发」",
    ),
    ResumeFieldDef(
        key="project_period",
        label="项目起止时间",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.PROJECT_EXPERIENCE,
    ),
    ResumeFieldDef(
        key="project_background",
        label="项目背景及目标",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.PROJECT_EXPERIENCE,
    ),
    ResumeFieldDef(
        key="project_content",
        label="核心工作内容与技术栈",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.PROJECT_EXPERIENCE,
    ),
    ResumeFieldDef(
        key="project_result",
        label="项目成果及效益",
        field_type=FieldType.STRING,
        required=True,
        module=ResumeModule.PROJECT_EXPERIENCE,
        description="量化或定性的项目成果，如性能提升指标、用户量等",
    ),
]

PROJECT_EXPERIENCE_FIELDS = [
    ResumeFieldDef(
        key="project_experience_list",
        label="项目经历",
        field_type=FieldType.LIST_DICT,
        required=True,
        module=ResumeModule.PROJECT_EXPERIENCE,
        description="一段或多段项目经历",
        is_repeating=True,
        sub_fields=PROJECT_SUB_FIELDS,
    ),
]

# --- Module 6: 专业技能与证书 ---

SKILLS_CERT_FIELDS = [
    ResumeFieldDef(
        key="hard_skills",
        label="硬技能",
        field_type=FieldType.DICT,
        required=False,
        module=ResumeModule.SKILLS_CERTIFICATES,
        description='专业技能及其熟练度，如 {"Java": "熟练", "Spring Boot": "精通"}',
        extraction_hint="用户提到的技术栈、编程语言、工具等，注意记录熟练程度",
    ),
    ResumeFieldDef(
        key="soft_skills",
        label="软技能",
        field_type=FieldType.LIST,
        required=False,
        module=ResumeModule.SKILLS_CERTIFICATES,
        description="沟通能力、团队协作、领导力等软技能",
    ),
    ResumeFieldDef(
        key="language_skills",
        label="语言能力",
        field_type=FieldType.DICT,
        required=False,
        module=ResumeModule.SKILLS_CERTIFICATES,
        description='语言能力及考试成绩，如 {"英语": "CET-6 550分", "日语": "N2"}',
    ),
    ResumeFieldDef(
        key="certificates",
        label="专业证书",
        field_type=FieldType.LIST,
        required=False,
        module=ResumeModule.SKILLS_CERTIFICATES,
        description="获得的专业证书及颁发机构和时间",
    ),
]

# --- Module 7: 其他个性化信息 ---

OTHER_INFO_FIELDS = [
    ResumeFieldDef(
        key="personal_summary",
        label="个人摘要/自我介绍",
        field_type=FieldType.STRING,
        required=False,
        module=ResumeModule.OTHER_INFO,
        description="用户对自己的简短介绍或职业摘要",
    ),
    ResumeFieldDef(
        key="portfolio_links",
        label="作品集/个人链接",
        field_type=FieldType.DICT,
        required=False,
        module=ResumeModule.OTHER_INFO,
        description='GitHub、个人博客、LinkedIn等链接，如 {"github": "url", "blog": "url"}',
    ),
    ResumeFieldDef(
        key="other_experiences",
        label="其他经历",
        field_type=FieldType.LIST,
        required=False,
        module=ResumeModule.OTHER_INFO,
        description="学生会、志愿者、社团活动等其他经历",
    ),
]


# ============================================================================
# Aggregate schema
# ============================================================================

# All fields flat list (for iteration / lookup)
ALL_FIELDS: List[ResumeFieldDef] = (
    JOB_INTENT_FIELDS
    + PERSONAL_INFO_FIELDS
    + EDUCATION_FIELDS
    + WORK_EXPERIENCE_FIELDS
    + PROJECT_EXPERIENCE_FIELDS
    + SKILLS_CERT_FIELDS
    + OTHER_INFO_FIELDS
)

# Index by key for O(1) lookup
FIELD_INDEX: Dict[str, ResumeFieldDef] = {f.key: f for f in ALL_FIELDS}

# Grouped by module — maps module -> list of fields
FIELDS_BY_MODULE: Dict[ResumeModule, List[ResumeFieldDef]] = {
    ResumeModule.JOB_INTENT: JOB_INTENT_FIELDS,
    ResumeModule.PERSONAL_INFO: PERSONAL_INFO_FIELDS,
    ResumeModule.EDUCATION: EDUCATION_FIELDS,
    ResumeModule.WORK_EXPERIENCE: WORK_EXPERIENCE_FIELDS,
    ResumeModule.PROJECT_EXPERIENCE: PROJECT_EXPERIENCE_FIELDS,
    ResumeModule.SKILLS_CERTIFICATES: SKILLS_CERT_FIELDS,
    ResumeModule.OTHER_INFO: OTHER_INFO_FIELDS,
}

# Required field keys (for missing-info detection)
REQUIRED_FIELD_KEYS: Set[str] = {f.key for f in ALL_FIELDS if f.required}


# ============================================================================
# Utility functions
# ============================================================================

def get_field_def(key: str) -> Optional[ResumeFieldDef]:
    """Get a field definition by its key.

    Args:
        key: Field key (e.g. "target_position", "school")

    Returns:
        ResumeFieldDef or None if not found
    """
    return FIELD_INDEX.get(key)


def get_module_fields(module: ResumeModule) -> List[ResumeFieldDef]:
    """Get all field definitions for a module.

    Args:
        module: The ResumeModule enum value

    Returns:
        List of field definitions
    """
    return FIELDS_BY_MODULE.get(module, [])


def get_required_fields() -> List[ResumeFieldDef]:
    """Get all required field definitions.

    Returns:
        List of required field definitions
    """
    return [f for f in ALL_FIELDS if f.required]


def get_required_keys() -> Set[str]:
    """Get all required field keys.

    Returns:
        Set of required field keys
    """
    return REQUIRED_FIELD_KEYS


def get_repeating_fields() -> List[ResumeFieldDef]:
    """Get all repeating (list-of-dict) field definitions.

    Returns:
        List of repeating field definitions
    """
    return [f for f in ALL_FIELDS if f.is_repeating]


def get_non_repeating_fields() -> List[ResumeFieldDef]:
    """Get all non-repeating field definitions (flat fields).

    Returns:
        List of non-repeating field definitions
    """
    return [f for f in ALL_FIELDS if not f.is_repeating]


def get_schema_summary() -> str:
    """Get a human-readable summary of the full schema.

    Returns:
        Multi-line string showing modules and their fields
    """
    lines = ["# 简历信息采集清单 Schema", ""]
    for module in MODULE_ORDER:
        label = MODULE_LABELS[module]
        fields = FIELDS_BY_MODULE[module]
        lines.append(f"## {label}")
        for f in fields:
            req = "必要" if f.required else "可选"
            if f.is_repeating:
                lines.append(f"  - [{req}] {f.key} ({f.label}) — 可多项")
                for sf in f.sub_fields:
                    s_req = "必要" if sf.required else "可选"
                    lines.append(f"    - [{s_req}] {sf.key} ({sf.label})")
            else:
                lines.append(f"  - [{req}] {f.key} ({f.label})")
        lines.append("")
    return "\n".join(lines)


def validate_resume_data(data: Dict[str, Any]) -> List[str]:
    """Validate resume data against schema, returning list of missing required keys.

    Args:
        data: The resume_data dict to validate

    Returns:
        List of missing required field key strings
    """
    missing = []
    for f in ALL_FIELDS:
        if not f.required:
            continue
        if f.is_repeating:
            # repeating fields should be a non-empty list
            val = data.get(f.key, [])
            if not val or not isinstance(val, list):
                missing.append(f.key)
        elif f.field_type in (FieldType.DICT,):
            val = data.get(f.key, {})
            if not val or not isinstance(val, dict):
                missing.append(f.key)
        else:
            val = data.get(f.key, "")
            if not val:
                missing.append(f.key)
    return missing


def build_flat_profile_for_llm(data: Dict[str, Any]) -> str:
    """Build a flat text summary of known resume data for LLM context.

    Args:
        data: The resume_data dict

    Returns:
        Formatted string like "已知简历信息：\n- 目标岗位：Java后端开发工程师\n..."
    """
    if not data:
        return "暂无已知简历信息"

    lines = ["已知简历信息："]
    for module in MODULE_ORDER:
        label = MODULE_LABELS[module]
        fields = FIELDS_BY_MODULE[module]
        module_items = []

        for f in fields:
            if f.is_repeating:
                entries = data.get(f.key, [])
                if entries:
                    vals = []
                    for entry in entries:
                        if isinstance(entry, dict):
                            parts = []
                            for sf in f.sub_fields:
                                sv = entry.get(sf.key, "")
                                if sv:
                                    parts.append(f"{sf.label}: {sv}")
                            if parts:
                                vals.append(" | ".join(parts))
                    if vals:
                        module_items.append(f"{f.label} ({len(vals)}段)")
                        for i, v in enumerate(vals):
                            module_items.append(f"  [{i+1}] {v}")
            else:
                val = data.get(f.key)
                if val:
                    if isinstance(val, list):
                        module_items.append(f"{f.label}：{'、'.join(val)}")
                    elif isinstance(val, dict):
                        inner = "、".join(f"{k}: {v}" for k, v in val.items())
                        module_items.append(f"{f.label}：{inner}")
                    else:
                        module_items.append(f"{f.label}：{val}")

        if module_items:
            lines.append(f"\n【{label}】")
            lines.extend(f"- {item}" for item in module_items)

    return "\n".join(lines)


def new_empty_resume_data() -> Dict[str, Any]:
    """Create a new empty resume_data dict with all schema keys.

    Returns:
        Dict with all schema keys, defaulting to empty string/list/dict
    """
    data: Dict[str, Any] = {}
    for f in ALL_FIELDS:
        if f.field_type in (FieldType.LIST,):
            data[f.key] = []
        elif f.field_type == FieldType.LIST_DICT:
            data[f.key] = []
        elif f.field_type == FieldType.DICT:
            data[f.key] = {}
        else:
            data[f.key] = ""
    return data
