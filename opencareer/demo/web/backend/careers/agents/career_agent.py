"""Career Agent with MCP tool integration.

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

try:
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import BaseTool
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from ..prompts.career import get_system_prompt, get_extraction_prompt

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

logger = logging.getLogger("careers.agents.career_agent")

SYSTEM_PROMPT = get_system_prompt() if LANGCHAIN_AVAILABLE else ""

# 智能断句系统提示词
SENTENCE_BREAK_PROMPT = """你是一个中文句子分割专家。你的任务是分析用户输入的文本，并在合适的位置分割成少量自然的句子或段落（建议3-4个气泡，不超过5个）。

分割原则（模仿微信聊天风格）：
1. 按语义和标点符号分割：在句号、问号、感叹号、换行符等自然的分割点分割
2. 控制气泡数量：尽量将相关内容合并，保持在3-4个气泡左右，不超过5个
3. 按长度控制：单个气泡不宜过短，建议100-300字左右
4. 保持上下文连贯：分割后的每个片段应该能独立理解

输出格式：
返回一个JSON数组，每个元素是一个句子或段落的内容。

示例：
输入："你好！我是小明。我想找工作。你能帮我吗？"
输出：["你好！我是小明。", "我想找工作。你能帮我吗？"]

输入："今天天气真不错啊！我想出去走走，你要不要一起？我们可以去公园，那边风景很好。"
输出：["今天天气真不错啊！我想出去走走，你要不要一起？", "我们可以去公园，那边风景很好。"]"""


def _sanitize_text(text: str) -> str:
    """Remove surrogate characters that cause UTF-8 encoding errors with DeepSeek API."""
    if not text:
        return text
    return text.encode("utf-8", errors="ignore").decode("utf-8")


class CareerAgent:
    """LangChain agent with MCP tool integration for career companion."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        mcp_url: str = "http://localhost:8001/mcp",
        memory_file: str = "career_memory.json",
        use_mcp: bool = True,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.mcp_url = mcp_url
        self.memory_file = memory_file
        self.use_mcp = use_mcp
        self._mcp_client = None
        self._tools: List[Any] = []

        if LANGCHAIN_AVAILABLE and self.api_key:
            self.llm = ChatOpenAI(
                temperature=0.7,
                model="deepseek-chat",
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
            )

            self.extraction_llm = ChatOpenAI(
                temperature=0.1,
                model="deepseek-chat",
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
            )

            self.sentence_break_llm = ChatOpenAI(
                temperature=0.1,
                model="deepseek-chat",
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
            )

            self.extraction_prompt = get_extraction_prompt()
            self.message_history = InMemoryChatMessageHistory()
        else:
            self.llm = None
            self.extraction_llm = None
            self.sentence_break_llm = None
            self.extraction_prompt = None
            self.message_history = None

        self.long_term_memory = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load memory file: {e}")
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
        try:
            Path(self.memory_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save memory: %s", e)

    async def connect_mcp(self) -> List[Any]:
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

            self._tools = await self._mcp_client.get_tools()
            logger.info("Loaded %s tools from MCP server: %s", len(self._tools), [t.name for t in self._tools])

        except Exception as e:
            logger.warning("Failed to connect to MCP server at %s: %s", self.mcp_url, e)
            logger.warning("Running without MCP tools — agent will use built-in knowledge only")
            self._tools = []

        return self._tools

    @property
    def tools(self) -> List[Any]:
        return self._tools

    async def chat(self, user_input: str) -> str:
        if not LANGCHAIN_AVAILABLE or not self.llm:
            return "抱歉，CareerAgent 所需的 LangChain 依赖尚未安装。"

        if self.use_mcp and not self._tools and not self._mcp_client:
            await self.connect_mcp()

        messages = self._build_messages(user_input)

        if self._tools:
            llm_with_tools = self.llm.bind_tools(self._tools)
        else:
            llm_with_tools = self.llm

        response = await llm_with_tools.ainvoke(messages)

        if response.tool_calls:
            tool_results = await self._execute_tool_calls(response.tool_calls)

            direct_outputs = []
            for tr in tool_results:
                try:
                    result_data = json.loads(tr.content)
                    # 处理 resume_skill 返回的格式
                    if result_data.get("ok"):
                        # 这是 resume_skill 的返回格式
                        resume = result_data.get("resume", {})
                        ats_report = result_data.get("ats_report", {})
                        notes = result_data.get("notes", [])
                        saved = result_data.get("saved", {})
                        
                        output_lines = []
                        output_lines.append("✅ 简历生成成功！\n")
                        
                        contact = resume.get("contact", {})
                        if contact:
                            output_lines.append(f"📋 简历信息：")
                            if contact.get("name"):
                                output_lines.append(f"   • 姓名：{contact['name']}")
                            if contact.get("title"):
                                output_lines.append(f"   • 岗位：{contact['title']}")
                        
                        if ats_report:
                            score = ats_report.get("score", 0)
                            output_lines.append(f"\n📊 ATS 评分：{score} 分")
                            if ats_report.get("issues"):
                                output_lines.append(f"\n⚠️  改进建议：")
                                for issue in ats_report["issues"]:
                                    output_lines.append(f"   • {issue}")
                        
                        if notes:
                            output_lines.append(f"\n📝 备注：")
                            for note in notes:
                                output_lines.append(f"   • {note}")
                        
                        if saved:
                            if saved.get("pdf"):
                                output_lines.append(f"\n💾 保存位置：")
                                output_lines.append(f"   • PDF：{saved['pdf']}")
                            if saved.get("json"):
                                output_lines.append(f"   • JSON：{saved['json']}")
                        
                        direct_outputs.append("\n".join(output_lines))
                    elif result_data.get("_type") == "markdown_skill":
                        # 兼容旧格式
                        output = result_data.get("output")
                        if output:
                            direct_outputs.append(output)
                except (json.JSONDecodeError, AttributeError):
                    pass

            if direct_outputs:
                response_text = "\n\n".join(direct_outputs)
            else:
                messages.append(response)
                for tr in tool_results:
                    messages.append(tr)

                final_response = await llm_with_tools.ainvoke(messages)
                response_text = final_response.content
        else:
            response_text = response.content

        response_text = _sanitize_text(response_text)

        if self.message_history:
            self.message_history.add_user_message(user_input)
            self.message_history.add_ai_message(response_text)

        self._update_memory(user_input, response_text)

        return response_text

    async def _split_into_sentences(self, text: str) -> List[str]:
        """使用 LLM 智能断句，模仿微信聊天风格（控制在3-4个气泡）"""
        if not self.sentence_break_llm:
            # 回退到简单的标点断句
            return self._simple_split(text)
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", SENTENCE_BREAK_PROMPT),
                ("user", "请分割以下文本：\n\n{text}")
            ])
            
            messages = prompt.format_messages(text=text)
            response = await self.sentence_break_llm.ainvoke(messages)
            
            extracted_text = _sanitize_text(response.content.strip())
            
            # 尝试提取 JSON
            if "```json" in extracted_text:
                extracted_text = extracted_text.split("```json")[1].split("```")[0].strip()
            elif "```" in extracted_text:
                extracted_text = extracted_text.split("```")[1].split("```")[0].strip()
            
            sentences = json.loads(extracted_text)
            
            if isinstance(sentences, list) and all(isinstance(s, str) for s in sentences):
                logger.info(f"智能断句成功: {len(sentences)} 个片段")
                
                # 如果气泡太多，进行合并
                if len(sentences) > 4:
                    sentences = self._merge_sentences(sentences, max_bubbles=4)
                
                return sentences
            
        except Exception as e:
            logger.warning(f"智能断句失败，使用回退方案: {e}")
        
        # 回退方案
        sentences = self._simple_split(text)
        if len(sentences) > 4:
            sentences = self._merge_sentences(sentences, max_bubbles=4)
        return sentences

    def _merge_sentences(self, sentences: List[str], max_bubbles: int = 4) -> List[str]:
        """合并过多的句子，控制气泡数量"""
        if len(sentences) <= max_bubbles:
            return sentences
        
        # 计算每组合并多少个
        merge_ratio = (len(sentences) + max_bubbles - 1) // max_bubbles
        
        result = []
        for i in range(0, len(sentences), merge_ratio):
            group = sentences[i:i + merge_ratio]
            merged = "".join(group)
            result.append(merged)
        
        logger.info(f"合并句子: {len(sentences)} -> {len(result)} 个气泡")
        return result

    def _simple_split(self, text: str) -> List[str]:
        """简单的标点断句回退方案"""
        if not text:
            return []
        
        # 按标点分割
        import re
        sentences = re.split(r'([。！？!?\n]+)', text)
        
        result = []
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            
            if sentence.strip():
                result.append(sentence.strip())
        
        # 如果没有分割出有效句子，返回整个文本
        if not result:
            return [text]
        
        return result

    async def stream_chat(self, user_input: str):
        """Stream agent response - 带有智能断句的微信风格聊天"""
        if not LANGCHAIN_AVAILABLE or not self.llm:
            yield "抱歉，CareerAgent 所需的 LangChain 依赖尚未安装。"
            return

        if self.use_mcp and not self._tools and not self._mcp_client:
            await self.connect_mcp()

        messages = self._build_messages(user_input)

        if self._tools:
            llm_with_tools = self.llm.bind_tools(self._tools)
        else:
            llm_with_tools = self.llm

        # 先完整生成响应
        response = await llm_with_tools.ainvoke(messages)

        full_response = ""

        if response.tool_calls:
            yield f"\n[调用工具: {', '.join(tc['name'] for tc in response.tool_calls)}]\n"

            tool_results = await self._execute_tool_calls(response.tool_calls)

            direct_outputs = []
            for tr in tool_results:
                try:
                    result_data = json.loads(tr.content)
                    # 处理 resume_skill 返回的格式
                    if result_data.get("ok"):
                        # 这是 resume_skill 的返回格式
                        resume = result_data.get("resume", {})
                        ats_report = result_data.get("ats_report", {})
                        notes = result_data.get("notes", [])
                        saved = result_data.get("saved", {})
                        
                        output_lines = []
                        output_lines.append("✅ 简历生成成功！\n")
                        
                        contact = resume.get("contact", {})
                        if contact:
                            output_lines.append(f"📋 简历信息：")
                            if contact.get("name"):
                                output_lines.append(f"   • 姓名：{contact['name']}")
                            if contact.get("title"):
                                output_lines.append(f"   • 岗位：{contact['title']}")
                        
                        if ats_report:
                            score = ats_report.get("score", 0)
                            output_lines.append(f"\n📊 ATS 评分：{score} 分")
                            if ats_report.get("issues"):
                                output_lines.append(f"\n⚠️  改进建议：")
                                for issue in ats_report["issues"]:
                                    output_lines.append(f"   • {issue}")
                        
                        if notes:
                            output_lines.append(f"\n📝 备注：")
                            for note in notes:
                                output_lines.append(f"   • {note}")
                        
                        if saved:
                            if saved.get("pdf"):
                                output_lines.append(f"\n💾 保存位置：")
                                output_lines.append(f"   • PDF：{saved['pdf']}")
                            if saved.get("json"):
                                output_lines.append(f"   • JSON：{saved['json']}")
                        
                        direct_outputs.append("\n".join(output_lines))
                    elif result_data.get("_type") == "markdown_skill":
                        # 兼容旧格式
                        output = result_data.get("output")
                        if output:
                            direct_outputs.append(output)
                except (json.JSONDecodeError, AttributeError):
                    pass

            if direct_outputs:
                response_text = "\n\n".join(direct_outputs)
                full_response = response_text
            else:
                messages.append(response)
                for tr in tool_results:
                    messages.append(tr)

                final_response = await llm_with_tools.ainvoke(messages)
                full_response = final_response.content
        else:
            full_response = response.content

        full_response = _sanitize_text(full_response)

        # 使用 LLM 智能断句（控制在3-4个气泡）
        sentences = await self._split_into_sentences(full_response)
        
        # 逐句发送，在句子之间发送断句信号
        for i, sentence in enumerate(sentences):
            yield sentence
            if i < len(sentences) - 1:
                # 发送断句信号
                yield "__FRAGMENT_BREAK__"
                logger.info(f"发送断句信号 (第 {i+1} 个片段)")

        # 保存到历史
        if self.message_history:
            self.message_history.add_user_message(user_input)
            if full_response:
                self.message_history.add_ai_message(full_response)

        # 更新记忆
        if full_response:
            self._update_memory(user_input, full_response)

    def _build_messages(self, user_input: str) -> list:
        user_input = _sanitize_text(user_input)
        user_context = self._get_user_context()

        system_content = SYSTEM_PROMPT
        if user_context:
            system_content += f"\n\n用户背景信息：\n{user_context}"

        messages = [SystemMessage(content=system_content)]

        if self.message_history:
            for msg in self.message_history.messages:
                messages.append(msg)

        messages.append(HumanMessage(content=user_input))

        return messages

    async def _execute_tool_calls(self, tool_calls: list) -> List[Any]:
        tool_name_map = {t.name for t in self._tools}
        results: List[Any] = []

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tool_id = tc["id"]

            tool = None
            for t in self._tools:
                if t.name == tool_name:
                    tool = t
                    break

            if tool:
                try:
                    if hasattr(tool, "ainvoke"):
                        result = await tool.ainvoke(tool_args)
                    else:
                        result = tool.invoke(tool_args)

                    content = str(result) if not isinstance(result, str) else result
                    logger.info("Tool '%s' executed successfully", tool_name)
                except Exception as e:
                    content = f"工具执行错误：{e}"
                    logger.error("Tool '%s' failed: %s", tool_name, e)
            else:
                content = f"工具 '{tool_name}' 未找到"
                logger.warning("Tool '%s' not found", tool_name)

            results.append(ToolMessage(content=content, tool_call_id=tool_id))

        return results

    def _get_user_context(self) -> str:
        parts = []

        if self.long_term_memory.get("user_info"):
            recent = list(self.long_term_memory["user_info"].values())[-5:]
            parts.append("用户信息：" + "；".join(recent))

        if self.long_term_memory.get("preferences"):
            recent = list(self.long_term_memory["preferences"].values())[-5:]
            parts.append("偏好：" + "；".join(recent))

        if self.long_term_memory.get("emotions"):
            recent = list(self.long_term_memory["emotions"].values())[-3:]
            parts.append("最近情感：" + "；".join(recent))

        if self.long_term_memory.get("important_events"):
            recent_events = self.long_term_memory["important_events"][-3:]
            events_str = "；".join([e["content"] if isinstance(e, dict) else e for e in recent_events])
            parts.append("重要事件：" + events_str)

        if self.long_term_memory.get("goals"):
            recent_goals = self.long_term_memory["goals"][-3:]
            goals_str = "；".join([e["content"] if isinstance(e, dict) else e for e in recent_goals])
            parts.append("用户目标：" + goals_str)

        if self.long_term_memory.get("last_interaction"):
            parts.append(f"上次交流：{self.long_term_memory['last_interaction']}")

        return "\n".join(parts) if parts else ""

    def _extract_important_info(self, user_input: str, ai_response: str) -> None:
        if not self.extraction_llm or not self.extraction_prompt:
            logger.warning("提取LLM或提示模板未初始化")
            return

        try:
            user_input = _sanitize_text(user_input)
            ai_response = _sanitize_text(ai_response)
            
            messages = self.extraction_prompt.format_messages(
                user_input=f"用户说：{user_input}\n\nAI回复：{ai_response}"
            )

            response = self.extraction_llm.invoke(messages)

            extracted_text = _sanitize_text(response.content.strip())
            logger.debug(f"原始提取结果: {extracted_text}")

            if "```json" in extracted_text:
                extracted_text = extracted_text.split("```json")[1].split("```")[0].strip()
            elif "```" in extracted_text:
                extracted_text = extracted_text.split("```")[1].split("```")[0].strip()

            extracted_data = json.loads(extracted_text)
            logger.debug(f"解析后的提取数据: {extracted_data}")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for info in extracted_data.get("user_info", []):
                if info and isinstance(info, str) and info.strip():
                    self.long_term_memory["user_info"][timestamp] = info.strip()
                    logger.info(f"提取到用户信息: {info}")

            for pref in extracted_data.get("preferences", []):
                if pref and isinstance(pref, str) and pref.strip():
                    self.long_term_memory["preferences"][timestamp] = pref.strip()
                    logger.info(f"提取到偏好: {pref}")

            for event in extracted_data.get("important_events", []):
                if event and isinstance(event, str) and event.strip():
                    self.long_term_memory["important_events"].append({
                        "timestamp": timestamp,
                        "content": event.strip(),
                    })
                    logger.info(f"提取到重要事件: {event}")

            for emotion in extracted_data.get("emotions", []):
                if emotion and isinstance(emotion, str) and emotion.strip():
                    self.long_term_memory["emotions"][timestamp] = emotion.strip()
                    logger.info(f"提取到情感: {emotion}")

            for goal in extracted_data.get("goals", []):
                if goal and isinstance(goal, str) and goal.strip():
                    self.long_term_memory["goals"].append({
                        "timestamp": timestamp,
                        "content": goal.strip(),
                    })
                    logger.info(f"提取到目标: {goal}")

            if any([
                extracted_data.get("user_info"),
                extracted_data.get("preferences"),
                extracted_data.get("important_events"),
                extracted_data.get("emotions"),
                extracted_data.get("goals"),
            ]):
                logger.info("提取到新信息：%s", extracted_data)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}，原始响应: {response.content if 'response' in locals() else 'N/A'}")
        except Exception as e:
            logger.warning(f"信息提取失败: {e}", exc_info=True)

    def _update_memory(self, user_input: str, response: str) -> None:
        if not response:
            logger.warning("响应内容为空，跳过记忆更新")
            return
            
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.long_term_memory["last_interaction"] = now

        self._extract_important_info(user_input, response)

        self._save_memory()
        logger.debug(f"记忆已更新，当前记忆统计: {self.get_memory_summary()}")

    def get_memory_summary(self) -> Dict[str, Any]:
        return {
            "user_info_count": len(self.long_term_memory.get("user_info", {})),
            "preferences_count": len(self.long_term_memory.get("preferences", {})),
            "emotions_count": len(self.long_term_memory.get("emotions", {})),
            "goals_count": len(self.long_term_memory.get("goals", [])),
            "last_interaction": self.long_term_memory.get("last_interaction", "无"),
        }

    def clear_memory(self) -> None:
        if self.message_history:
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
