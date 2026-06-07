import asyncio
import logging
import sys
import json
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO)


async def test_mcp_resume_skill():
    """测试 MCP resume_skill 是否能正常调用"""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        print("=" * 60)
        print("连接 MCP 服务器...")
        print("=" * 60)
        
        client = MultiServerMCPClient({
            "opencareer": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8001",
            }
        })
        
        try:
            print("\n获取可用工具列表...")
            tools = await client.get_tools()
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
            
            # 调用 resume_skill 的 generate 动作
            result = await client.call_tool(
                "resume_skill",
                {
                    "action": "generate",
                    "name": "张三",
                    "target_role": "Python后端开发工程师",
                    "job_type": "社招",
                    "industry": "互联网",
                    "city": "北京",
                    "experience_years": "3年",
                    "phone": "13800138000",
                    "email": "zhangsan@example.com",
                    "education": "北京科技大学 / 计算机科学与技术 / 本科 / 2021.07",
                    "summary": "Python后端开发，3年工作经验，擅长 FastAPI、Django 等框架，熟悉 MySQL、Redis、Docker 等技术栈，有分布式系统开发经验。",
                    "experiences": [
                        "负责电商平台后端 API 开发，日均处理订单 10 万+，响应时间优化至 200ms 以内。",
                        "使用 Redis 实现热点数据缓存，QPS 提升 300%，降低数据库压力 50%。",
                        "主导用户中心重构，采用微服务架构，系统可用性提升至 99.9%。"
                    ],
                    "projects": [
                        "订单管理系统：负责核心模块开发，支持多渠道接入，日订单量超 50 万。",
                        "用户画像系统：基于大数据分析用户行为，实现个性化推荐，转化率提升 15%。"
                    ],
                    "skills": [
                        "Python", "FastAPI", "Django", "MySQL", "Redis", "Docker", "Git", "Linux"
                    ],
                    "certifications": [
                        "阿里云 ACE 认证", "PMP 项目管理认证"
                    ]
                }
            )
            
            print("\n" + "=" * 60)
            print("调用结果：")
            print("=" * 60)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # 检查结果是否成功
            if result.get("ok"):
                print("\n✅ resume_skill 调用成功！")
                print(f"  - PDF 已生成: {result.get('pdf_path')}")
                print(f"  - JSON 已保存: {result.get('json_path')}")
                print(f"  - ATS 分数: {result.get('ats_report', {}).get('score')}分")
                return True
            else:
                print(f"\n❌ resume_skill 调用失败: {result.get('error')}")
                return False
                
        finally:
            await client.close()
            
    except Exception as e:
        import traceback
        print(f"\n❌ 异常: {type(e).__name__}: {e}")
        print("\n堆栈信息:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_mcp_resume_skill())
    sys.exit(0 if success else 1)
