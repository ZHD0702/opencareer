import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent  # F:\opencareer\web\backend\test_integration.py -> F:\opencareer
sys.path.insert(0, str(project_root))

print("Project root:", project_root)
print("Python path:", sys.path[:3])

try:
    from opencareer.agents.career_agent import CareerAgent
    print("✅ CareerAgent imported successfully!")
    
    print("Testing configuration...")
    from careers_config import config
    print(f"Agent type: {config.AGENT_TYPE}")
    print(f"Use MCP: {config.USE_MCP}")
    print(f"DeepSeek API Key (prefix): {config.DEEPSEEK_API_KEY[:10] if config.DEEPSEEK_API_KEY else 'Not set'}...")
    
    print("\n✅ Integration OK! Backend should start without issues.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
