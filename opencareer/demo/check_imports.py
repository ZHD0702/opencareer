#!/usr/bin/env python3
"""
Import check for OpenCareer demo.

This script verifies that all key components can be imported without errors.
It's useful for verifying the project structure before running tests.
"""

import sys
import importlib

def check_import(module_path, class_name=None):
    """Check if a module or class can be imported."""
    try:
        module = importlib.import_module(module_path)
        if class_name:
            getattr(module, class_name)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    """Check all key imports."""
    print("OpenCareer Demo - Import Check")
    print("=" * 60)

    checks = [
        # Core modules
        ("opencareer.agents.base_agent", "BaseAgent"),
        ("opencareer.agents.brain.brain_agent", "BrainAgent"),
        ("opencareer.agents.work_agent.work_agent", "WorkAgent"),
        ("opencareer.agents.emotion_agent.emotion_agent", "EmotionAgent"),
        ("opencareer.agents.log_agent.log_agent", "LogAgent"),

        # SKILLs
        ("opencareer.skills.base_skill", "BaseSkill"),
        ("opencareer.skills.skill_registry", "SkillRegistry"),
        ("opencareer.skills.learning_plan.skill", "LearningPlanSkill"),
        ("opencareer.skills.mock_interview.skill", "MockInterviewSkill"),
        ("opencareer.skills.emotion_support.skill", "EmotionSupportSkill"),

        # MCP
        ("opencareer.mcp.server", "MCPServer"),
        ("opencareer.mcp.skill_loader", "SkillLoader"),

        # Memory
        ("opencareer.memory.memory_manager", "MemoryManager"),
        ("opencareer.memory.vector_memory", "ChromaVectorMemory"),
        ("opencareer.memory.structured_memory", "SQLiteStructuredMemory"),

        # Scheduler
        ("opencareer.scheduler.task_scheduler", "TaskScheduler"),
    ]

    all_passed = True
    failed_checks = []

    for module_path, class_name in checks:
        print(f"Checking {module_path}.{class_name}... ", end="")
        passed, error = check_import(module_path, class_name)

        if passed:
            print("OK")
        else:
            print("FAIL")
            print(f"  Error: {error}")
            all_passed = False
            failed_checks.append((module_path, class_name, error))

    print("\n" + "=" * 60)

    if all_passed:
        print("SUCCESS: All imports passed!")
        print("\nKey statistics:")
        print(f"  - Agents: 4 (Brain, Work, Emotion, Log)")
        print(f"  - SKILLs: 3 (Learning Plan, Mock Interview, Emotion Support)")
        print(f"  - Memory: 2 (Vector, Structured) + Manager")
        print(f"  - MCP: Server + Loader")
        print(f"  - Scheduler: TaskScheduler")
    else:
        print(f"FAILED: {len(failed_checks)} import(s) failed")
        print("\nFailed imports:")
        for module_path, class_name, error in failed_checks:
            print(f"  - {module_path}.{class_name}: {error}")

        print("\nTroubleshooting tips:")
        print("  1. Ensure you're in the demo directory")
        print("  2. Check that all Python files exist")
        print("  3. Verify there are no syntax errors in the files")
        print("  4. Run: python -m py_compile path/to/file.py to check syntax")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())