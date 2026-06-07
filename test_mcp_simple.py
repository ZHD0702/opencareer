import asyncio
import logging
import sys
import json
sys.path.insert(0, ".")

logging.basicConfig(level=logging.DEBUG)


async def test_simple_mcp():
    """简单的 MCP 客户端测试"""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        print("=" * 60)
        print("测试 MCP 连接 (使用 stdio)...")
        print("=" * 60)
        
        # 使用 stdio 直接启动 MCP 服务器（这样更简单）
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "opencareer.mcp.server"]
        )
        
        async with stdio_client(server_params) as (read, write):
            session = ClientSession(read, write)
            
            async with session:
                print("\n初始化会话...")
                await session.initialize()
                
                print("\n获取可用工具...")
                tools_result = await session.list_tools()
                tools = tools_result.tools
                
                print(f"\n✅ 加载了 {len(tools)} 个工具：")
                for t in tools:
                    print(f"  - {t.name}: {t.description[:60]}...")
                
                # 查找 resume_skill
                resume_tool = None
                for t in tools:
                    if t.name == "resume_skill":
                        resume_tool = t
                        break
                
                if not resume_tool:
                    print("\n❌ 未找到 resume_skill 工具！")
                    return False
                
                print("\n" + "=" * 60)
                print("测试调用 resume_skill (generate)...")
                print("=" * 60)
                
                # 调用 resume_skill
                result = await session.call_tool(
                    "resume_skill",
                    {
                        "action": "generate",
                        "name": "李四",
                        "target_role": "前端开发工程师",
                        "job_type": "校招",
                        "city": "上海",
                        "phone": "13900139000",
                        "email": "lisi@example.com",
                        "education": "复旦大学 / 软件工程 / 本科 / 2025.06",
                        "experiences": [
                            "在校期间完成多个前端项目，熟练使用 React、Vue 等框架。"
                        ],
                        "skills": [
                            "JavaScript", "TypeScript", "React", "Vue", "HTML", "CSS"
                        ]
                    }
                )
                
                print("\n" + "=" * 60)
                print("调用结果：")
                print("=" * 60)
                print(json.dumps(result.content[0].text if result.content else result, 
                               ensure_ascii=False, indent=2))
                
                return True
                
    except Exception as e:
        import traceback
        print(f"\n❌ 异常: {type(e).__name__}: {e}")
        print("\n堆栈信息:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simple_mcp())
    sys.exit(0 if success else 1)
