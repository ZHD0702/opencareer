"""快速测试 resume-skill 是否正常工作"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from opencareer.agents.career_agent import CareerAgent

async def test_resume_skill():
    print("=" * 60)
    print("测试 CareerAgent + resume-skill")
    print("=" * 60)
    
    # 创建 agent，启用 MCP
    agent = CareerAgent(
        use_mcp=True,
        mcp_url="http://localhost:8001/mcp"
    )
    
    # 连接 MCP
    print("\n正在连接 MCP 服务器...")
    await agent.connect_mcp()
    
    if agent.tools:
        print(f"\n✅ 成功加载 {len(agent.tools)} 个工具:")
        for t in agent.tools:
            print(f"  - {t.name}")
        
        # 检查 resume_skill 是否存在
        has_resume = any(t.name == "resume_skill" for t in agent.tools)
        if has_resume:
            print("\n✅ resume_skill 已启用！")
            print("\n" + "=" * 60)
            print("你现在可以通过以下方式使用 resume-skill：")
            print("=" * 60)
            print("1. CLI 方式: python main.py")
            print("2. Web 方式: 启动 web/backend 然后使用前端界面")
            print("\n在聊天中输入类似：")
            print('  - "帮我生成一份简历"')
            print('  - "优化我的简历"')
            print('  - "检查我的简历ATS兼容性"')
            print('  - "导出为PDF"')
            print("=" * 60)
            return True
        else:
            print("\n❌ 未找到 resume_skill 工具")
            return False
    else:
        print("\n❌ 未加载任何工具，MCP 连接可能失败")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_resume_skill())
    sys.exit(0 if success else 1)
