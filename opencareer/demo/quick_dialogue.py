#!/usr/bin/env python3
"""
OpenCareer 快速对话测试
使用真实DeepSeek API密钥的交互式对话
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, Any

# Windows编码处理
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 添加当前目录和opencareer目录到Python路径（prompts包在opencareer下）
_demo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _demo_dir)
sys.path.insert(0, os.path.join(_demo_dir, "opencareer"))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("quick_dialogue")

async def setup_real_agents():
    """设置真实智能体（使用DeepSeek API）"""
    from opencareer.agents.brain.brain_agent import BrainAgent
    from opencareer.agents.work_agent.work_agent import WorkAgent
    from opencareer.agents.emotion_agent.emotion_agent import EmotionAgent
    from opencareer.agents.log_agent.log_agent import LogAgent

    # 获取API密钥
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    if not deepseek_api_key or deepseek_api_key == "your_deepseek_api_key_here":
        logger.error("未找到有效的DeepSeek API密钥！")
        logger.error("请在.env文件中设置DEEPSEEK_API_KEY")
        logger.error("当前密钥: %s", deepseek_api_key[:20] if deepseek_api_key else "None")
        return None

    logger.info("使用真实DeepSeek API密钥: %s...", deepseek_api_key[:10])

    # 创建共享对话上下文
    from opencareer.agents.conversation_context import ConversationContext
    ctx = ConversationContext(user_id="quick_dialogue_user")

    # 创建共享 LLM 客户端
    from opencareer.agents.llm_client import create_llm_client
    llm_client = create_llm_client()
    if llm_client:
        logger.info("共享 LLM 客户端已创建")
    else:
        logger.warning("LLM 客户端创建失败，智能体将使用备用模式")

    # 创建大脑智能体
    brain = BrainAgent(llm_client=llm_client, deepseek_api_key=deepseek_api_key, conversation_context=ctx)
    logger.info("大脑智能体已创建（真实AI模式）")

    # 创建其他智能体（共享 LLM 客户端）
    work_agent = WorkAgent(llm_client=llm_client, mcp_server_url="http://localhost:8000", conversation_context=ctx)
    emotion_agent = EmotionAgent(llm_client=llm_client, mcp_server_url="http://localhost:8000", conversation_context=ctx)
    log_agent = LogAgent(memory_manager=None)

    # 模拟智能体注册表
    class MockAgentRegistry:
        def get_agent(self, name):
            agents = {
                'work_agent': work_agent,
                'emotion_agent': emotion_agent,
                'log_agent': log_agent
            }
            return agents.get(name)

    brain.agent_registry = MockAgentRegistry()
    logger.info("所有智能体设置完成")
    return brain

async def process_request(brain_agent, user_input: str) -> Dict[str, Any]:
    """处理用户请求"""
    try:
        result = await brain_agent.process_user_request(user_input)

        # 获取完整的需求分析和目标智能体
        demand_analysis = result.get("demand_analysis", {})
        target_agent = result.get("target_agent", "work_agent")
        response = result.get("response", "")

        # 路由到下游智能体
        downstream = brain_agent.agent_registry.get_agent(target_agent) if brain_agent.agent_registry else None
        if downstream:
            context = {"demand_analysis": demand_analysis}
            agent_result = await downstream.process_user_request(user_input, context)
            return {
                "demand_analysis": demand_analysis,
                "demand_type": demand_analysis.get("demand_type", "unknown"),
                "agent_used": target_agent,
                "skill_used": agent_result.get("skill_used", ""),
                "response": agent_result.get("response", response),
                "status": agent_result.get("status", "success")
            }

        # 没有下游智能体，返回大脑的响应
        return {
            "demand_analysis": demand_analysis,
            "demand_type": demand_analysis.get("demand_type", "unknown"),
            "agent_used": "brain",
            "response": response or "抱歉，我暂时无法处理这个请求。",
            "status": result.get("status", "success")
        }
    except Exception as e:
        logger.error(f"处理错误: {e}")
        return {"error": str(e), "status": "error"}

async def interactive_conversation():
    """交互式对话"""
    print("\n" + "="*70)
    print("OpenCareer 快速对话测试 - 真实AI模式")
    print("="*70)
    print("\n您好！我是OpenCareer求职助手。")
    print("我可以帮助您解决职业发展、面试准备和情感支持等问题。")

    # 设置智能体
    print("\n正在初始化智能体系统...")
    brain_agent = await setup_real_agents()
    if not brain_agent:
        print("初始化失败！请检查API密钥配置。")
        return

    print("智能体系统就绪！")

    # 主动开场：智能体先输出开场白
    greeting = await brain_agent.get_opening_greeting()
    print(f"\n[OpenCareer] {greeting}")

    print("\n您可以开始对话了（输入 'quit' 退出）:")
    print("-"*70)

    conversation_count = 0
    while True:
        try:
            # 获取输入
            prompt = f"[{conversation_count + 1}] 您: "
            user_input = input(prompt).strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n感谢使用OpenCareer！祝您求职顺利！")
                break

            if user_input.lower() in ['help', '?']:
                print("\n帮助:")
                print("  • 职业发展: '我需要学习Python'")
                print("  • 面试准备: '我明天有面试'")
                print("  • 情感支持: '我感到很紧张'")
                print("  • 混合需求: '我面试很紧张，需要准备'")
                print("  输入 'quit' 退出")
                continue

            # 处理请求
            print(f"\n[处理中] OpenCareer正在分析您的需求...")
            response = await process_request(brain_agent, user_input)

            # 显示响应
            demand_type = response.get('demand_type', 'unknown')
            agent_used = response.get('agent_used', 'brain')
            agent_label = {"brain": "大脑", "work_agent": "工作智能体", "emotion_agent": "情感智能体"}

            skill_used = response.get('skill_used', 'none')
            print(f"\n[分析] 需求类型: {demand_type}")
            print(f"[调度] 智能体: {agent_label.get(agent_used, agent_used)}")
            print(f"[技能] {skill_used}")
            print(f"[响应] {response.get('response', '无响应')}")
            print(f"[状态] {response.get('status', '未知')}")
            print("-"*70)

            conversation_count += 1

        except KeyboardInterrupt:
            print("\n\n对话结束。再见！")
            break
        except EOFError:
            print("\n\n输入结束。再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")

def main():
    """主函数"""
    print("启动OpenCareer快速对话测试...")

    try:
        asyncio.run(interactive_conversation())
    except KeyboardInterrupt:
        print("\n\n测试中断")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        print(f"\n启动失败: {e}")

if __name__ == "__main__":
    main()