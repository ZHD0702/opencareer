"""
AI Career Companion — CLI entry point.

Uses CareerAgent (LangChain + MCP) for career companion conversations.
The MCP server must be running separately, or use --no-mcp for offline mode.

Usage:
    python main.py              # Connect to MCP server
    python main.py --no-mcp     # Offline mode (LLM-only, no tools)
    python main.py --stream     # Streaming response mode
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

# Load .env before importing project modules
load_dotenv()

from opencareer.agents.career_agent import CareerAgent

logger = logging.getLogger("opencareer.main")

BANNER = r"""
  ██████╗ ██████╗ ███████╗███╗   ██╗
  ██╔═══██╗██╔══██╗██╔════╝████╗  ██║
  ██║   ██║██████╔╝█████╗  ██╔██╗ ██║
  ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║
  ╚██████╔╝██║     ███████╗██║ ╚████║
   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝
  ██████╗ █████╗ ██████╗ ███████╗███████╗██████╗
  ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗
  ██║     ███████║██████╔╝█████╗  █████╗  ██████╔╝
  ██║     ██╔══██║██╔══██╗██╔══╝  ██╔══╝  ██╔══██╗
  ╚██████╗██║  ██║██║  ██║███████╗███████╗██║  ██║
   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
                AI CAREER COMPANION
"""

HELP_TEXT = """
可用命令:
  quit / exit  — 退出程序
  memory       — 查看记忆摘要
  detail       — 查看详细记忆内容
  clear        — 清除所有记忆
  /stream      — 切换流式输出模式
"""


class CLI:
    """Command-line interface for CareerAgent."""

    def __init__(self, agent: CareerAgent, stream_mode: bool = False):
        self.agent = agent
        self.stream_mode = stream_mode

    async def run(self):
        print(BANNER)
        print(f"MCP 状态: {'已连接' if self.agent.tools else '未连接（离线模式）'}")
        print(f"已加载工具: {[t.name for t in self.agent.tools] if self.agent.tools else '无'}")
        print(f"流式输出: {'开启' if self.stream_mode else '关闭'}")
        print(HELP_TEXT)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ("quit", "exit"):
                    print("再见！祝你求职顺利~")
                    break

                if user_input.lower() == "memory":
                    self._show_memory_summary()
                    continue

                if user_input.lower() == "detail":
                    self._show_memory_detail()
                    continue

                if user_input.lower() == "clear":
                    self.agent.clear_memory()
                    print("记忆已清除")
                    continue

                if user_input.lower() == "/stream":
                    self.stream_mode = not self.stream_mode
                    print(f"流式输出: {'开启' if self.stream_mode else '关闭'}")
                    continue

                if not user_input:
                    continue

                # Process user input
                if self.stream_mode:
                    print("Agent: ", end="", flush=True)
                    async for token in self.agent.stream_chat(user_input):
                        print(token, end="", flush=True)
                    print()
                else:
                    print("Agent: ", end="", flush=True)
                    response = await self.agent.chat(user_input)
                    print(response)

            except KeyboardInterrupt:
                print("\n\n再见！祝你求职顺利~")
                break
            except Exception as e:
                logger.exception(f"Error processing input: {e}")
                print(f"\n出错了: {e}")

    def _show_memory_summary(self):
        summary = self.agent.get_memory_summary()
        print("\n记忆摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    def _show_memory_detail(self):
        memory = self.agent.long_term_memory
        print("\n详细记忆:")
        for category, items in memory.items():
            print(f"\n  [{category}]")
            if isinstance(items, dict) and items:
                for ts, val in list(items.items())[-5:]:
                    print(f"    [{ts}] {val}")
            elif isinstance(items, list) and items:
                for item in items[-5:]:
                    print(f"    - {item}")
            else:
                print("    (暂无)")


async def main():
    parser = argparse.ArgumentParser(description="AI Career Companion CLI")
    parser.add_argument("--no-mcp", action="store_true", help="Offline mode (no MCP tools)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    parser.add_argument("--mcp-url", default="http://localhost:8000/mcp", help="MCP server URL")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    agent = CareerAgent(
        use_mcp=not args.no_mcp,
        mcp_url=args.mcp_url,
    )

    # Connect to MCP (if enabled) — load tools
    if agent.use_mcp:
        print(f"正在连接 MCP 服务器 ({args.mcp_url})...")
        await agent.connect_mcp()
        if agent.tools:
            print(f"成功加载 {len(agent.tools)} 个工具")
        else:
            print("MCP 服务器未连接，将使用离线模式")

    cli = CLI(agent, stream_mode=args.stream)
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
