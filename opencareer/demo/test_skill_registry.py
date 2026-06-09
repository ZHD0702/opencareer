"""
Test script for Resume Cn Career SKILL integration with SkillRegistry.
"""

import asyncio
import sys
from pathlib import Path

# Add demo directory to path
sys.path.insert(0, str(Path(__file__).parent))

from opencareer.skills.skill_registry import get_global_registry
from opencareer.skills.resume_cn_career import create_resume_cn_career_skill


async def test_registry_integration():
    """Test integration with SkillRegistry."""
    print("=" * 60)
    print("Testing Resume Cn Career SKILL with SkillRegistry")
    print("=" * 60)
    
    # Get global registry
    registry = get_global_registry()
    
    # Create and register skill
    print("\n1. Creating and registering skill...")
    skill = create_resume_cn_career_skill()
    registry.register(skill)
    print(f"   ✓ Skill registered: {skill.metadata.name}")
    
    # List skills
    print("\n2. Listing registered skills...")
    skills_list = registry.list_skills()
    for sk in skills_list:
        print(f"   - {sk['name']} (v{sk['version']}) - {sk['description'][:50]}...")
    
    # Get skill
    print("\n3. Retrieving skill from registry...")
    retrieved_skill = registry.get("resume_cn_career")
    if retrieved_skill:
        print(f"   ✓ Retrieved: {retrieved_skill.metadata.name}")
    else:
        print("   ✗ Failed to retrieve skill")
        return
    
    # Test through registry
    print("\n4. Executing skill through registry...")
    test_input = {
        "action": "generate",
        "name": "李四",
        "target_role": "产品经理",
        "job_type": "社招",
        "industry": "互联网",
        "city": "上海",
        "experience_years": "3年",
        "education": "复旦大学 / 市场营销 / 硕士 / 2021"
    }
    
    result = await registry.execute("resume_cn_career", test_input)
    print(f"   ✓ Registry execute result: {'OK' if result['ok'] else 'ERROR'}")
    if result['ok']:
        print(f"   ✓ Resume name: {result['resume']['contact']['name']}")
        print(f"   ✓ ATS Score: {result['ats_report']['score']}")
    
    # Cleanup
    print("\n5. Cleaning up...")
    await registry.cleanup_all()
    print("   ✓ Cleanup complete")
    
    print("\n" + "=" * 60)
    print("Registry integration tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_registry_integration())

