"""
Career Agent with MCP tool integration.

Uses langchain-mcp-adapters' MultiServerMCPClient to connect to the
OpenCareer MCP server and load tools as LangChain BaseTools.
The agent uses DeepSeek via ChatOpenAI with tool calling.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from opencareer.prompts.career import get_system_prompt, get_extraction_prompt

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)

logger = logging.getLogger("opencareer.agents.career_agent")

SYSTEM_PROMPT = get_system_prompt()


class CareerAgent:
    """LangChain agent with MCP tool integration for career companion.

    Architecture:
        User Input → CareerAgent.chat()
          → LLM decides: respond directly OR call MCP tool
          → If tool called: execute via MCP → LLM formulates final response
          → Response returned with tool call trace
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        mcp_url: str = "http://localhost:8001/mcp",
        memory_file: str = "career_memory.json",
        use_mcp: bool = True,
    ):
        """
        Args:
            api_key: DeepSeek API key
            mcp_url: MCP server Streamable HTTP endpoint
            memory_file: Long-term memory JSON file path
            use_mcp: Whether to connect to MCP server (set False for testing)
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.mcp_url = mcp_url
        self.memory_file = memory_file
        self.use_mcp = use_mcp
        self._mcp_client = None
        self._tools: List[BaseTool] = []

        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="deepseek-chat",
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
        )

        # 用于信息提取的LLM（temperature更低以获得更准确的提取）
        self.extraction_llm = ChatOpenAI(
            temperature=0.1,
            model="deepseek-chat",
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
        )

        # 创建信息提取的提示模板
        self.extraction_prompt = get_extraction_prompt()

        # Short-term memory (in-memory chat history)
        self.message_history = InMemoryChatMessageHistory()

        # Long-term memory (persisted to JSON)
        self.long_term_memory = self._load_memory()

    # ------------------------------------------------------------------
    # Memory persistence
    # ------------------------------------------------------------------

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_file):
            try:
                import json
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "user_info": {},
            "preferences": {},
            "emotions": {},
            "goals": [],
            "important_events": [],
            "conversation_summary": [],
            "last_interaction": None,
        }

    def _save_memory(self) -> None:
        import json
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")

    # ------------------------------------------------------------------
    # MCP client management
    # ------------------------------------------------------------------

    async def connect_mcp(self) -> List[BaseTool]:
        """Connect to the MCP server and load tools.

        Returns:
            List of LangChain BaseTools loaded from MCP server
        """
        if not self.use_mcp:
            logger.info("MCP is disabled (use_mcp=False)")
            return []

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            self._mcp_client = MultiServerMCPClient({
                "opencareer": {
                    "transport": "streamable_http",
                    "url": self.mcp_url,
                }
            })

            # Load all tools from the MCP server
            self._tools = await self._mcp_client.get_tools()
            logger.info(f"Loaded {len(self._tools)} tools from MCP server: "
                        f"{[t.name for t in self._tools]}")

        except Exception as e:
            logger.warning(f"Failed to connect to MCP server at {self.mcp_url}: {e}")
            logger.warning("Running without MCP tools — agent will use built-in knowledge only")
            self._tools = []

        return self._tools

    @property
    def tools(self) -> List[BaseTool]:
        return self._tools

    # ------------------------------------------------------------------
    # Chat — main entry point
    # ------------------------------------------------------------------

    async def chat(self, user_input: str) -> str:
        """Process user input and return agent response.

        The agent uses LLM with tool calling:
        1. LLM receives user input + system prompt + tool definitions
        2. LLM decides whether to call a tool or respond directly
        3. If tool called, result is fed back to LLM for final response
        """
        # Ensure MCP is connected
        if self.use_mcp and not self._tools and not self._mcp_client:
            await self.connect_mcp()

        # Build messages
        messages = self._build_messages(user_input)

        # Call LLM with tool binding
        if self._tools:
            llm_with_tools = self.llm.bind_tools(self._tools)
        else:
            llm_with_tools = self.llm

        response = await llm_with_tools.ainvoke(messages)

        # Handle tool calls
        if response.tool_calls:
            # Execute tool calls
            tool_results = await self._execute_tool_calls(response.tool_calls)

            # Feed tool results back to LLM for final response
            messages.append(response)
            for tr in tool_results:
                messages.append(tr)

            final_response = await llm_with_tools.ainvoke(messages)
            response_text = final_response.content
        else:
            response_text = response.content

        # Store in chat history
        self.message_history.add_user_message(user_input)
        self.message_history.add_ai_message(response_text)

        # Update long-term memory
        self._update_memory(user_input, response_text)

        return response_text

    async def stream_chat(self, user_input: str):
        """Stream agent response token by token.

        Yields:
            Tokens from the LLM response (str). If tool calls are made,
            yields a summary marker then the final response tokens.
        """
        if self.use_mcp and not self._tools and not self._mcp_client:
            await self.connect_mcp()

        messages = self._build_messages(user_input)

        if self._tools:
            llm_with_tools = self.llm.bind_tools(self._tools)
        else:
            llm_with_tools = self.llm

        response = await llm_with_tools.ainvoke(messages)

        # Handle tool calls
        if response.tool_calls:
            yield f"\n[调用工具: {', '.join(tc['name'] for tc in response.tool_calls)}]\n"

            tool_results = await self._execute_tool_calls(response.tool_calls)
            messages.append(response)
            for tr in tool_results:
                messages.append(tr)

            # Stream final response
            async for chunk in llm_with_tools.astream(messages):
                if chunk.content:
                    yield chunk.content
        else:
            response_text = response.content or ""
            # For non-streaming initial response, yield all at once
            yield response_text

        # Store in history
        self.message_history.add_user_message(user_input)
        if response.content:
            self.message_history.add_ai_message(response.content)

        self._update_memory(user_input, response.content or "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, user_input: str) -> list:
        """Build the message list for LLM invocation."""
        user_context = self._get_user_context()

        system_content = SYSTEM_PROMPT
        if user_context:
            system_content += f"\n\n用户背景信息：\n{user_context}"

        messages = [SystemMessage(content=system_content)]

        # Add chat history
        for msg in self.message_history.messages:
            messages.append(msg)

        # Add current user input
        messages.append(HumanMessage(content=user_input))

        return messages

    async def _execute_tool_calls(self, tool_calls: list) -> List[ToolMessage]:
        """Execute MCP tool calls and return ToolMessages."""
        tool_name_map = {t.name: t for t in self._tools}
        results: List[ToolMessage] = []

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tool_id = tc["id"]

            tool = tool_name_map.get(tool_name)
            if tool:
                try:
                    # Execute the tool (supports both sync and async)
                    if hasattr(tool, "ainvoke"):
                        result = await tool.ainvoke(tool_args)
                    else:
                        result = tool.invoke(tool_args)

                    content = str(result) if not isinstance(result, str) else result
                    logger.info(f"Tool '{tool_name}' executed successfully")
                except Exception as e:
                    content = f"工具执行错误: {e}"
                    logger.error(f"Tool '{tool_name}' failed: {e}")
            else:
                content = f"工具 '{tool_name}' 未找到"
                logger.warning(f"Tool '{tool_name}' not found")

            results.append(ToolMessage(content=content, tool_call_id=tool_id))

        return results

    def _get_user_context(self) -> str:
        """Extract user context from long-term memory for the system prompt."""
        parts = []

        if self.long_term_memory.get("user_info"):
            recent = list(self.long_term_memory["user_info"].values())[-5:]
            parts.append("用户信息: " + "; ".join(recent))

        if self.long_term_memory.get("preferences"):
            recent = list(self.long_term_memory["preferences"].values())[-5:]
            parts.append("偏好: " + "; ".join(recent))

        if self.long_term_memory.get("emotions"):
            recent = list(self.long_term_memory["emotions"].values())[-3:]
            parts.append("最近情绪: " + "; ".join(recent))

        if self.long_term_memory.get("important_events"):
            recent_events = self.long_term_memory["important_events"][-3:]
            events_str = "; ".join([e["content"] if isinstance(e, dict) else e for e in recent_events])
            parts.append("重要事件: " + events_str)

        if self.long_term_memory.get("goals"):
            recent_goals = self.long_term_memory["goals"][-3:]
            goals_str = "; ".join([e["content"] if isinstance(e, dict) else e for e in recent_goals])
            parts.append("用户目标: " + goals_str)

        if self.long_term_memory.get("last_interaction"):
            parts.append(f"上次交流: {self.long_term_memory['last_interaction']}")

        return "\n".join(parts) if parts else ""

    def _extract_important_info(self, user_input: str, ai_response: str) -> None:
        """使用LLM从对话中提取重要信息"""
        try:
            # 构建提取提示
            messages = self.extraction_prompt.format_messages(user_input=user_input)

            # 调用LLM提取信息
            response = self.extraction_llm.invoke(messages)

            # 解析JSON响应
            extracted_text = response.content.strip()

            # 尝试提取JSON（处理可能的markdown代码块）
            if "```json" in extracted_text:
                extracted_text = extracted_text.split("```json")[1].split("```")[0].strip()
            elif "```" in extracted_text:
                extracted_text = extracted_text.split("```")[1].split("```")[0].strip()

            extracted_data = json.loads(extracted_text)

            # 保存提取的信息
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 保存用户信息
            for info in extracted_data.get("user_info", []):
                if info:
                    self.long_term_memory["user_info"][timestamp] = info

            # 保存偏好
            for pref in extracted_data.get("preferences", []):
                if pref:
                    self.long_term_memory["preferences"][timestamp] = pref

            # 保存重要事件
            for event in extracted_data.get("important_events", []):
                if event:
                    self.long_term_memory["important_events"].append({
                        "timestamp": timestamp,
                        "content": event
                    })

            # 保存情感状态
            for emotion in extracted_data.get("emotions", []):
                if emotion:
                    self.long_term_memory["emotions"][timestamp] = emotion

            # 保存目标
            for goal in extracted_data.get("goals", []):
                if goal:
                    self.long_term_memory["goals"].append({
                        "timestamp": timestamp,
                        "content": goal
                    })

            # 如果提取到了信息，打印日志
            if any([
                extracted_data.get("user_info"),
                extracted_data.get("preferences"),
                extracted_data.get("important_events"),
                extracted_data.get("emotions"),
                extracted_data.get("goals")
            ]):
                print(f"\n[记忆更新] 提取到新信息:")
                for key, values in extracted_data.items():
                    if values:
                        print(f"  - {key}: {values}")

        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
        except Exception as e:
            logger.warning(f"信息提取失败: {e}")

    def _update_memory(self, user_input: str, response: str) -> None:
        """更新长期记忆 - 使用LLM提取信息"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.long_term_memory["last_interaction"] = now

        # 使用LLM提取重要信息
        self._extract_important_info(user_input, response)

        self._save_memory()

    def get_memory_summary(self) -> Dict[str, int]:
        return {
            "用户信息条数": len(self.long_term_memory.get("user_info", {})),
            "偏好记录数": len(self.long_term_memory.get("preferences", {})),
            "情感记录数": len(self.long_term_memory.get("emotions", {})),
            "目标数": len(self.long_term_memory.get("goals", [])),
            "最后交互": self.long_term_memory.get("last_interaction", "无"),
        }

    def clear_memory(self) -> None:
        self.message_history.clear()
        self.long_term_memory = {
            "user_info": {},
            "preferences": {},
            "emotions": {},
            "goals": [],
            "important_events": [],
            "last_interaction": None,
        }
        self._save_memory()


# ------------------------------------------------------------------
# Quick test entry
# ------------------------------------------------------------------

async def test_agent():
    """Test the agent without MCP (offline mode)."""
    logging.basicConfig(level=logging.INFO)

    agent = CareerAgent(use_mcp=False)
    print("Career Agent (offline mode) ready.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit"):
                break

            response = await agent.chat(user_input)
            print(f"Agent: {response}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_agent())
