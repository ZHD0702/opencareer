"""Career Agent 配置文件 - 使用外面的 opencareer 包"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 先加载外面项目根目录的 .env 文件
_project_root = Path(__file__).resolve().parent.parent.parent  # F:\opencareer\web\backend\careers_config.py -> F:\opencareer
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# 再加载 web/backend 目录的 .env（如果有）
_local_env_path = Path(__file__).resolve().parent / ".env"
if _local_env_path.exists():
    load_dotenv(_local_env_path)


class CareersConfig:
    """Career Agent 配置类"""
    
    # Agent 配置 - 默认启用 CareerAgent
    AGENT_TYPE = os.getenv("CAREER_AGENT_TYPE", "career")
    USE_MCP = os.getenv("CAREER_USE_MCP", "false").lower() == "true"
    
    # MCP 配置
    MCP_URL = os.getenv("CAREER_MCP_URL", "http://localhost:8001/mcp")
    MCP_HOST = os.getenv("CAREER_MCP_HOST", "127.0.0.1")
    MCP_PORT = int(os.getenv("CAREER_MCP_PORT", "8001"))
    
    # 记忆配置
    MEMORY_DIR = Path(os.getenv("CAREER_MEMORY_DIR", "./data/memory"))
    MEMORY_FILE = os.getenv("CAREER_MEMORY_FILE", "career_memory.json")
    
    # DeepSeek 配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    @classmethod
    def get_memory_path(cls) -> Path:
        """获取记忆文件完整路径"""
        cls.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        return cls.MEMORY_DIR / cls.MEMORY_FILE
    
    @classmethod
    def is_career_agent_enabled(cls) -> bool:
        """检查是否启用 Career Agent"""
        return cls.AGENT_TYPE == "career"
    
    @classmethod
    def validate_config(cls) -> tuple[bool, list[str]]:
        """验证配置是否有效"""
        errors = []
        
        if cls.is_career_agent_enabled():
            if not cls.DEEPSEEK_API_KEY:
                errors.append("DEEPSEEK_API_KEY is required for Career Agent")
        
        return len(errors) == 0, errors


config = CareersConfig
