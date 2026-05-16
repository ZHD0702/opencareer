import sys
import asyncio
import logging
sys.path.insert(0, ".")

logging.basicConfig(level=logging.DEBUG)

async def test():
    from langchain_mcp_adapters.client import MultiServerMCPClient
    
    client = MultiServerMCPClient({
        "opencareer": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",
        }
    })
    
    try:
        tools = await client.get_tools()
        print(f"Tools loaded: {len(tools)}")
        for t in tools:
            print(f"  - {t.name}")
    except Exception as e:
        import traceback
        print(f"Exception: {type(e).__name__}: {e}")
        traceback.print_exc()

asyncio.run(test())
