#!/usr/bin/env python3
"""
LLM 适配器测试脚本
测试 Phase 1 任务 3: LLM 适配器
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.registry import get_llm_adapter
import asyncio


def test_llm():
    print("=" * 60)
    print("Phase 1 任务 3: LLM 适配器测试")
    print("=" * 60)
    
    # 1. 测试适配器初始化
    print("\n1. 测试 LLM 适配器初始化...")
    try:
        llm = get_llm_adapter("deepseek")
        print(f"   ✅ 适配器初始化成功: {llm.model}")
        print(f"   📍 API 端点: {llm.api_url}")
    except ValueError as e:
        print(f"   ⚠️  缺少配置: {e}")
        print("   💡 请在 .env 文件中设置 DEEPSEEK_API_KEY")
        return False
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        return False
    
    # 2. 测试同步调用
    print("\n2. 测试同步调用 (invoke)...")
    try:
        messages = [
            {"role": "user", "content": "你好，请介绍一下自己"}
        ]
        response = asyncio.run(llm.invoke(messages))
        print(f"   ✅ 同步调用成功")
        print(f"   🤖 AI 回复: {response[:100]}...")
    except Exception as e:
        print(f"   ❌ 同步调用失败: {e}")
        return False
    
    # 3. 测试流式调用
    print("\n3. 测试流式调用 (stream)...")
    try:
        messages = [
            {"role": "user", "content": "用一句话介绍一下 AI 助手"}
        ]
        print("   📤 流式输出: ", end="", flush=True)
        tokens = []
        async def stream_test():
            async for token in llm.stream(messages):
                print(token, end="", flush=True)
                tokens.append(token)
        asyncio.run(stream_test())
        print()
        print(f"   ✅ 流式调用成功: 共 {len(tokens)} 个 token")
    except Exception as e:
        print(f"   ❌ 流式调用失败: {e}")
        return False
    
    # 4. 测试对话历史
    print("\n4. 测试对话历史...")
    try:
        messages = [
            {"role": "user", "content": "我喜欢编程"},
            {"role": "assistant", "content": "太好了！编程是一项很有价值的技能。"},
            {"role": "user", "content": "请基于我的爱好给出建议"}
        ]
        response = asyncio.run(llm.invoke(messages))
        print(f"   ✅ 对话历史测试成功")
        print(f"   🤖 AI 回复: {response[:100]}...")
    except Exception as e:
        print(f"   ❌ 对话历史测试失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ 所有 LLM 适配器测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_llm()
    sys.exit(0 if success else 1)
