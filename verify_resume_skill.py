"""验证 resume-skill 配置是否正确"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("验证 resume-skill 配置")
print("=" * 60)

# 检查配置
import os
print("\n1. 检查环境变量:")
print(f"   CAREER_USE_MCP =", os.getenv("CAREER_USE_MCP"))
print(f"   CAREER_MCP_URL =", os.getenv("CAREER_MCP_URL"))
print(f"   CAREER_MCP_PORT =", os.getenv("CAREER_MCP_PORT"))
print(f"   DEEPSEEK_API_KEY =", "已设置" if os.getenv("DEEPSEEK_API_KEY") else "未设置")

# 检查必要模块
print("\n2. 检查模块是否可以导入:")
try:
    from opencareer.mcp.server import mcp
    print("   ✅ opencareer.mcp.server")
except Exception as e:
    print(f"   ❌ opencareer.mcp.server:", e)

try:
    from opencareer.mcp.tools.resume_tool import resume_skill
    print("   ✅ opencareer.mcp.tools.resume_tool")
except Exception as e:
    print(f"   ❌ opencareer.mcp.tools.resume_tool:", e)

try:
    from opencareer.agents.career_agent import CareerAgent
    print("   ✅ opencareer.agents.career_agent")
except Exception as e:
    print(f"   ❌ opencareer.agents.career_agent:", e)

# 检查 web 后端配置
print("\n3. 检查 Web 后端配置:")
try:
    sys.path.insert(0, str(Path(__file__).parent / "web" / "backend"))
    from careers_config import config
    print(f"   ✅ config.USE_MCP = {config.USE_MCP}")
    print(f"   ✅ config.MCP_URL = {config.MCP_URL}")
except Exception as e:
    print(f"   ❌ Web 配置:", e)

print("\n" + "=" * 60)
print("配置完成！现在你可以：")
print("=" * 60)
print("\n1. 使用 CLI 测试:")
print("   python main.py")
print("\n2. 启动 Web 后端:")
print("   cd web/backend")
print("   python -m uvicorn main:app --reload --port 8080")
print("\n3. resume-skill 的功能包括:")
print("   - 生成简历")
print("   - 优化简历")
print("   - ATS 检查")
print("   - PDF 导出")
print("=" * 60)
