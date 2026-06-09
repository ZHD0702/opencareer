"""
Example: LangChain + MCP Integration for OpenCareer.

This script demonstrates how to use the LangChain agent with
OpenCareer SKILLs via MCP protocol.
"""

import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from opencareer.langchain_mcp import create_opencareer_agent, OpenCareerAgent


async def example_1_basic_chat():
    """Example 1: Basic chat with the agent."""
    print("\n" + "="*60)
    print("Example 1: Basic Chat")
    print("="*60)

    # Create agent
    agent = create_opencareer_agent(
        memory_file=Path("./data/agent_memory.json")
    )

    # Initialize
    await agent.initialize()

    # Chat
    response = await agent.chat("你好，我想找工作，能帮我吗？")
    print(f"\n用户: 你好，我想找工作，能帮我吗？")
    print(f"助手: {response['response']}")

    # Save state
    agent.save_state()


async def example_2_profile_and_skills():
    """Example 2: Using user profile and direct skill execution."""
    print("\n" + "="*60)
    print("Example 2: Profile and Skills")
    print("="*60)

    # Create agent
    agent = create_opencareer_agent()
    await agent.initialize()

    # Update user profile
    agent.update_user_profile(
        name="张三",
        target_role="软件工程师",
        experience_years="3年",
        education="北京大学 计算机科学 本科"
    )

    print(f"\n用户资料: {agent.get_user_profile()}")

    # Direct skill execution (resume)
    print("\n--- 直接调用简历生成技能 ---")
    resume_result = await agent.execute_skill_direct(
        "resume_cn_career",
        {
            "action": "generate",
            "name": "张三",
            "target_role": "软件工程师",
            "experience_years": "3年",
            "education": "北京大学 计算机科学 本科",
            "experiences": [
                "在字节跳动实习3个月，开发用户增长功能",
                "获得国家奖学金"
            ]
        }
    )

    print(f"简历生成结果: {resume_result.get('ok', False)}")
    if resume_result.get('ok'):
        print(f"简历摘要: {resume_result.get('resume', {}).get('summary', '')[:100]}...")


async def example_3_multi_turn_conversation():
    """Example 3: Multi-turn conversation with memory."""
    print("\n" + "="*60)
    print("Example 3: Multi-turn Conversation")
    print("="*60)

    # Create agent with persistent memory
    agent = create_opencareer_agent(
        memory_file=Path("./data/conversation_memory.json")
    )
    await agent.initialize()

    # Set initial profile
    agent.update_user_profile(
        name="李四",
        target_role="产品经理"
    )

    # First turn
    print("\n--- 第一轮对话 ---")
    response1 = await agent.chat("我最近面试总是失败，感觉很沮丧")
    print(f"用户: 我最近面试总是失败，感觉很沮丧")
    print(f"助手: {response1['response'][:200]}...")

    # Second turn (should remember context)
    print("\n--- 第二轮对话 ---")
    response2 = await agent.chat("你觉得我应该从哪些方面提升？")
    print(f"用户: 你觉得我应该从哪些方面提升？")
    print(f"助手: {response2['response'][:200]}...")

    # Save
    agent.save_state()


async def interactive_chat():
    """Interactive chat mode."""
    print("\n" + "="*60)
    print("OpenCareer 交互式聊天模式")
    print("="*60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'profile' 查看用户资料")
    print("输入 'save' 保存当前状态\n")

    # Create agent
    agent = create_opencareer_agent(
        memory_file=Path("./data/interactive_memory.json")
    )
    await agent.initialize()

    while True:
        try:
            user_input = input("\n你: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("再见！祝你求职顺利！")
                agent.save_state()
                break

            if user_input.lower() == "profile":
                profile = agent.get_user_profile()
                print(f"\n用户资料: {profile}")
                continue

            if user_input.lower() == "save":
                agent.save_state()
                print("状态已保存！")
                continue

            # Chat with agent
            response = await agent.chat(user_input)

            if response["success"]:
                print(f"\n助手: {response['response']}")
            else:
                print(f"\n抱歉，出错了: {response.get('error', '未知错误')}")

        except KeyboardInterrupt:
            print("\n\n再见！")
            agent.save_state()
            break
        except Exception as e:
            print(f"\n错误: {e}")


async def main():
    """Main function to run examples."""
    print("\nOpenCareer LangChain + MCP 集成示例")
    print("="*60)

    # Create data directory
    Path("./data").mkdir(exist_ok=True)

    # Run examples
    try:
        await example_1_basic_chat()
        # await example_2_profile_and_skills()
        # await example_3_multi_turn_conversation()

        # Interactive mode
        print("\n" + "="*60)
        print("是否进入交互式聊天模式？(y/n)")
        choice = input().strip().lower()
        if choice == "y":
            await interactive_chat()

    except Exception as e:
        print(f"\n示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
