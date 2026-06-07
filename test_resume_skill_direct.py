import sys
import json
import os
sys.path.insert(0, ".")

print("=" * 60)
print("测试 resume_skill 功能（直接调用，不通过 MCP）")
print("=" * 60)

# 直接导入 resume_skill
from opencareer.mcp.tools.resume_tool import resume_skill

print("\n" + "=" * 60)
print("测试 1: 生成简历草稿 (action=generate)")
print("=" * 60)

result = resume_skill(
    action="generate",
    name="王五",
    target_role="产品经理",
    job_type="社招",
    industry="互联网",
    city="深圳",
    experience_years="5年",
    phone="13700137000",
    email="wangwu@example.com",
    education="深圳大学 / 工商管理 / 本科 / 2020.06",
    summary="5年互联网产品经验，主导过多个百万 DAU 产品，擅长用户增长、需求分析和产品规划。",
    experiences=[
        "负责社交产品核心功能设计，用户数从 50 万增长到 300 万，留存提升 25%。",
        "主导电商产品改版，优化购买转化路径，GMV 增长 40%。",
        "建立产品需求分析和迭代流程，提升团队效率 30%。"
    ],
    projects=[
        "用户成长体系设计：搭建积分、等级、任务系统，提升用户粘性和活跃度。",
        "AI 推荐系统产品设计：和算法团队协作，推出个性化推荐，点击率提升 35%。"
    ],
    skills=[
        "Axure", "Figma", "SQL", "Python", "数据分析", "用户研究", "A/B测试"
    ],
    certifications=[
        "PMP 项目管理认证", "NPDP 新产品开发认证"
    ]
)

print("\n调用结果：")
print(json.dumps(result, ensure_ascii=False, indent=2))

if result.get("ok"):
    print("\n✅ resume_skill 核心功能正常！")
    print(f"  - PDF 已生成: {result.get('pdf_path')}")
    print(f"  - JSON 已保存: {result.get('json_path')}")
    print(f"  - ATS 分数: {result.get('ats_report', {}).get('score')}分")
    
    # 检查文件是否真的生成了
    pdf_path = result.get("pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        print(f"  ✅ PDF 文件存在: {pdf_path}")
    else:
        print(f"  ⚠️  PDF 文件可能不存在")
    
    json_path = result.get("json_path")
    if json_path and os.path.exists(json_path):
        print(f"  ✅ JSON 文件存在: {json_path}")
    else:
        print(f"  ⚠️  JSON 文件可能不存在")
else:
    print(f"\n❌ 调用失败: {result.get('error')}")

print("\n" + "=" * 60)
print("测试 2: ATS 检查 (action=ats_check)")
print("=" * 60)

# 一个简单的 JD 进行测试
test_jd = """
岗位：产品经理
要求：
- 3年以上互联网产品经验
- 有用户增长、电商或社交产品经验
- 熟悉数据分析、A/B测试
- 熟练使用 Axure、Figma 等工具
- 良好的沟通协调能力
"""

result2 = resume_skill(
    action="ats_check",
    name="王五",
    target_role="产品经理",
    job_type="社招",
    city="深圳",
    phone="13700137000",
    email="wangwu@example.com",
    education="深圳大学 / 工商管理 / 本科 / 2020.06",
    experiences=[
        "负责社交产品核心功能设计，用户数从 50 万增长到 300 万，留存提升 25%。"
    ],
    skills=["Axure", "Figma", "数据分析", "A/B测试"],
    jd=test_jd
)

print("\nATS 检查结果：")
print(json.dumps(result2, ensure_ascii=False, indent=2))

if result2.get("ok"):
    print("\n✅ ATS 检查功能正常！")
else:
    print(f"\n❌ ATS 检查失败: {result2.get('error')}")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)

if result.get("ok") and result2.get("ok"):
    print("✅ resume_skill 所有核心功能正常！")
    print("\n可以通过 MCP 集成到 Agent 中使用。")
else:
    print("❌ resume_skill 存在问题，需要排查。")
