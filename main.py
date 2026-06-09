"""
OpenCareer — CLI entry point.

Uses CareerAgent (LangChain + MCP) for career companion conversations.
Automatically starts the MCP server as a subprocess, or use --no-mcp for offline mode.

Usage:
    python main.py              # Auto-start MCP + connect
    python main.py --no-mcp     # Offline mode (LLM-only, no tools)
    python main.py --stream     # Streaming response mode
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

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
                OPENCareer CLI
"""

HELP_TEXT = """
可用命令:
  quit / exit  — 退出程序
  memory       — 查看记忆摘要
  detail       — 查看详细记忆内容
  clear        — 清除所有记忆
  match        — 岗位匹配推荐（优先 AI 决策，回退本地匹配）
  /stream      — 切换流式输出模式
"""

def _normalize_memory_for_matcher(memory: dict) -> dict:
    """适配 CareerAgent 的记忆格式到匹配器兼容的格式。"""
    normalized = dict(memory)

    # CareerAgent stores goals as list of {timestamp, content}; matcher expects dict
    goals = memory.get("goals", [])
    if isinstance(goals, list):
        normalized["goals"] = {}
        for item in goals:
            if isinstance(item, dict):
                ts = item.get("timestamp", "")
                content = item.get("content", str(item))
                normalized["goals"][ts] = content
            else:
                normalized["goals"][str(time.time())] = str(item)

    return normalized


class CLI:
    """Command-line interface for CareerAgent."""

    def __init__(self, agent: CareerAgent, stream_mode: bool = False,
                 memory_file: str = "career_memory.json",
                 excel_path: str = "岗位匹配数据.xlsx"):
        self.agent = agent
        self.stream_mode = stream_mode
        self.memory_file = memory_file
        self.excel_path = excel_path

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

                if user_input.lower() == "match":
                    await self._run_job_match()
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

    async def _run_job_match(self):
        """岗位匹配：优先 AI 决策，回退本地关键词匹配。"""
        # 确保 Excel 数据文件存在
        if not os.path.exists(self.excel_path):
            print(f"\n[岗位匹配] 未找到岗位数据文件: {self.excel_path}")
            print("请将岗位匹配数据文件放到项目根目录。")
            return

        # 确保有用户画像
        memory = self.agent.long_term_memory
        if not memory.get("user_info") and not memory.get("goals") and not memory.get("preferences"):
            print("\n[岗位匹配] 暂无用户画像信息，请先跟助手聊聊天，让它了解你的背景。")
            return

        # 将 CareerAgent 记忆写入临时文件供匹配器读取
        match_memory = _normalize_memory_for_matcher(memory)
        temp_memory_path = self.memory_file + ".match_tmp"
        with open(temp_memory_path, "w", encoding="utf-8") as f:
            json.dump(match_memory, f, ensure_ascii=False, indent=2)

        try:
            # 阶段 1: 尝试 AI 匹配
            print("\n[岗位匹配] 尝试 AI 智能匹配...")
            try:
                from career_matcher_ai import ai_match_career
                # ai_match_career is synchronous, run in thread
                report = await asyncio.to_thread(
                    ai_match_career,
                    excel_path=self.excel_path,
                    memory_path=temp_memory_path,
                    coarse_n=30,
                    final_n=5,
                )
                print()
                print(report)
                return
            except Exception as e:
                print(f"  AI 匹配不可用: {e}")
                print("  回退到本地关键词匹配...\n")

            # 阶段 2: 回退到本地匹配
            from career_matcher import print_recommendations
            await asyncio.to_thread(
                print_recommendations,
                excel_path=self.excel_path,
                memory_path=temp_memory_path,
                top_n=10,
            )

        finally:
            # 清理临时文件
            try:
                os.remove(temp_memory_path)
            except OSError:
                pass


def _start_mcp_server(mcp_port: int = 8001) -> subprocess.Popen | None:
    """启动 MCP 服务器子进程。若端口已被占用则跳过。"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        sock.connect(("127.0.0.1", mcp_port))
        sock.close()
        print(f"MCP 服务器已在端口 {mcp_port} 运行")
        return None
    except (ConnectionRefusedError, OSError, socket.timeout):
        sock.close()

    project_root = str(Path(__file__).resolve().parent)
    print("正在启动 MCP 服务器...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "opencareer.mcp.server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=project_root,
    )

    # 等待子进程启动并就绪
    for _ in range(20):
        time.sleep(0.3)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            if "10048" in stderr or "Address already in use" in stderr:
                print("MCP 服务器已在运行（端口被占用）")
                return None
            print(f"MCP 服务器启动失败:\n{stderr}")
            return None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            sock.connect(("127.0.0.1", mcp_port))
            sock.close()
            print("MCP 服务器已就绪")
            return proc
        except (ConnectionRefusedError, OSError, socket.timeout):
            pass

    # 已达到最大等待时间，检查进程是否还活着
    if proc.poll() is None:
        print("MCP 服务器已就绪（通过进程状态）")
        return proc

    print("MCP 服务器启动超时")
    return None


async def main():
    parser = argparse.ArgumentParser(description="OpenCareer CLI")
    parser.add_argument("--no-mcp", action="store_true", help="Offline mode (no MCP tools)")
    parser.add_argument("--no-auto-mcp", action="store_true",
                        help="Don't auto-start MCP server (requires manual start)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    parser.add_argument("--mcp-url", default="http://localhost:8001/mcp", help="MCP server URL")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # 自动启动 MCP 服务器
    mcp_proc = None
    auto_mcp = not args.no_mcp and not args.no_auto_mcp
    if auto_mcp:
        mcp_proc = _start_mcp_server()

    agent = CareerAgent(
        use_mcp=not args.no_mcp,
        mcp_url=args.mcp_url,
    )

    # Connect to MCP (if enabled)
    if agent.use_mcp:
        print(f"正在连接 MCP 服务器 ({args.mcp_url})...")
        await agent.connect_mcp()
        if agent.tools:
            print(f"成功加载 {len(agent.tools)} 个工具")
        else:
            print("MCP 服务器未连接，将使用离线模式")

    try:
        cli = CLI(agent, stream_mode=args.stream)
        await cli.run()
    finally:
        # 清理 MCP 子进程
        if mcp_proc and mcp_proc.poll() is None:
            print("正在关闭 MCP 服务器...")
            mcp_proc.terminate()
            try:
                mcp_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mcp_proc.kill()
            print("MCP 服务器已关闭")


if __name__ == "__main__":
    asyncio.run(main())
