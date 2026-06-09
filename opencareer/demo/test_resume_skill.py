"""
Test script for Resume Cn Career SKILL integration.
"""

import asyncio
import sys
from pathlib import Path

# Add demo directory to path
sys.path.insert(0, str(Path(__file__).parent))

from opencareer.skills.resume_cn_career import ResumeCnCareerSkill


async def test_basic_functionality():
    """Test basic functionality of the resume skill."""
    print("=" * 60)
    print("Testing Resume Cn Career SKILL")
    print("=" * 60)
    
    # Create skill instance
    print("\n1. Creating skill instance...")
    skill = ResumeCnCareerSkill()
    print(f"   ✓ Skill created: {skill.metadata.name}")
    print(f"   ✓ Version: {skill.metadata.version}")
    print(f"   ✓ Category: {skill.metadata.category}")
    
    # Initialize skill
    print("\n2. Initializing skill...")
    await skill.initialize()
    print("   ✓ Skill initialized successfully")
    
    # Test 1: Generate resume
    print("\n3. Testing resume generation...")
    test_input = {
        "action": "generate",
        "name": "张三",
        "target_role": "软件工程师",
        "job_type": "校招",
        "industry": "互联网",
        "city": "北京",
        "experience_years": "应届生",
        "phone": "13800138000",
        "email": "zhangsan@example.com",
        "education": "北京大学 / 计算机科学与技术 / 本科 / 2024",
        "summary": "热爱编程，在校成绩优异，有丰富的项目经验",
        "experiences": [
            "在字节跳动实习3个月，参与抖音推荐系统优化",
            "使用Python和Spark处理海量数据，提升推荐准确率5%",
            "获得国家奖学金和校级三好学生"
        ],
        "projects": [
            "毕业设计：基于深度学习的图像识别系统",
            "使用TensorFlow实现，准确率达95%"
        ],
        "skills": ["Python", "Java", "TensorFlow", "Spark", "机器学习"],
        "certifications": ["计算机等级考试三级"],
        "bilingual": False
    }
    
    result = await skill.execute(test_input)
    print(f"   ✓ Execute result: {'OK' if result['ok'] else 'ERROR'}")
    
    if result['ok']:
        print(f"   ✓ Action: {result['action']}")
        print(f"   ✓ ATS Score: {result['ats_report']['score']}")
        if result['notes']:
            print(f"   ✓ Notes: {result['notes']}")
        
        resume = result['resume']
        print(f"   ✓ Resume name: {resume['contact']['name']}")
        print(f"   ✓ Resume title: {resume['contact']['title']}")
        print(f"   ✓ Number of sections: {len(resume['sections'])}")
    
    # Test 2: ATS check
    print("\n4. Testing ATS check...")
    ats_input = {
        "action": "ats_check",
        "name": "张三",
        "target_role": "软件工程师",
        "education": "北京大学 / 计算机 / 本科",
        "jd": """
        职位要求：
        - 熟悉 Python、Java 等编程语言
        - 了解机器学习、深度学习
        - 有 Spark、TensorFlow 经验
        - 良好的沟通能力
        """
    }
    
    ats_result = await skill.execute(ats_input)
    print(f"   ✓ ATS check result: {'OK' if ats_result['ok'] else 'ERROR'}")
    if ats_result['ok']:
        print(f"   ✓ ATS Score: {ats_result['ats_report']['score']}")
        if 'keyword_coverage_percent' in ats_result['ats_report']:
            print(f"   ✓ Keyword coverage: {ats_result['ats_report']['keyword_coverage_percent']}%")
    
    # Cleanup
    print("\n5. Cleaning up...")
    await skill.cleanup()
    print("   ✓ Cleanup complete")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())

