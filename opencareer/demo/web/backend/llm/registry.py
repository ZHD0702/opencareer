from .base import LLMAdapter
from .deepseek import DeepSeekAdapter
import os
from dotenv import load_dotenv

load_dotenv()

LLM_REGISTRY = {
    "deepseek": DeepSeekAdapter,
}


def _lazy_import_openai():
    """懒加载 OpenAI 适配器"""
    try:
        from .openai import OpenAIAdapter
        return OpenAIAdapter
    except ImportError:
        return None


def _lazy_import_claude():
    """懒加载 Claude 适配器"""
    try:
        from .claude import ClaudeAdapter
        return ClaudeAdapter
    except ImportError:
        return None


def get_llm_adapter(name: str = None) -> LLMAdapter:
    """
    获取指定的 LLM 适配器
    
    Args:
        name: LLM 提供者名称 (deepseek, openai, claude)
    
    Returns:
        LLMAdapter 实例
    """
    name = name or os.getenv("LLM_PROVIDER", "deepseek")
    
    # 尝试懒加载适配器
    if name == "openai":
        adapter_class = _lazy_import_openai()
        if adapter_class is None:
            raise ImportError("OpenAI adapter not available. Install with: pip install openai")
    elif name == "claude":
        adapter_class = _lazy_import_claude()
        if adapter_class is None:
            raise ImportError("Claude adapter not available. Install with: pip install anthropic")
    else:
        adapter_class = LLM_REGISTRY.get(name)
    
    if adapter_class is None:
        raise ValueError(f"LLM adapter '{name}' not implemented yet. Available: {list(LLM_REGISTRY.keys())} + openai, claude")
    
    return adapter_class()


def list_available_llms() -> list:
    """列出所有可用的 LLM 提供者"""
    available = ["deepseek"]
    
    # 检查 OpenAI
    if _lazy_import_openai() is not None and os.getenv("OPENAI_API_KEY"):
        available.append("openai")
    
    # 检查 Claude
    if _lazy_import_claude() is not None and os.getenv("ANTHROPIC_API_KEY"):
        available.append("claude")
    
    return available
