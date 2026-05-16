"""Career Agent 提示词加载器。"""

import yaml
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

_PROMPTS_DIR = Path(__file__).parent
_cache: dict = {}


def _load(name: str) -> dict:
    if name not in _cache:
        with open(_PROMPTS_DIR / f"{name}.yaml", "r", encoding="utf-8") as f:
            _cache[name] = yaml.safe_load(f)
    return _cache[name]


def get_system_prompt() -> str:
    return _load("system_prompt")["system_prompt"]


def get_system_prompt_with_context(user_context: str) -> str:
    data = _load("system_prompt")
    prompt = data["system_prompt"]
    if user_context:
        prompt += f"\n\n{data['context_template']}".format(user_context=user_context)
    return prompt


def get_extraction_prompt() -> ChatPromptTemplate:
    data = _load("extraction")
    return ChatPromptTemplate.from_messages([
        ("system", data["system_template"]),
        ("user", data["user_template"]),
    ])
