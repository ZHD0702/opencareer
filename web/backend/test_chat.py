import sys
sys.path.insert(0, '.')
import asyncio

async def test():
    from agent_factory import get_agent_factory
    factory = get_agent_factory()
    print('创建 agent...')
    agent = factory.create_agent('career', mcp_url='http://localhost:8001', use_mcp=True, memory_file='test_memory.json')
    print('Agent type:', type(agent).__name__)
    print('有 stream_chat:', hasattr(agent, 'stream_chat'))
    
    print('\n测试 chat...')
    try:
        tokens = []
        async for token in agent.stream_chat('你好'):
            tokens.append(token)
            if len(tokens) < 20:
                print(f'  token: {repr(token)[:80]}')
        print(f'\n收到 {len(tokens)} 个 tokens')
        result = ''.join(tokens)
        print(f'完整内容: {result[:500]}')
    except Exception as e:
        import traceback
        print(f'聊天错误: {type(e).__name__}: {e}')
        traceback.print_exc()

asyncio.run(test())
