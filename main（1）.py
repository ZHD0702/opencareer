# emotional_chatbot.py
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class ExtractedInfo(BaseModel):
    """提取的信息结构"""
    user_info: List[str] = Field(default_factory=list, description="用户的个人信息，如职业、居住地、家庭状况等")
    preferences: List[str] = Field(default_factory=list, description="用户的偏好和兴趣爱好")
    important_events: List[str] = Field(default_factory=list, description="重要事件、日期、纪念日等")
    emotions: List[str] = Field(default_factory=list, description="用户表达的情感状态")
    goals: List[str] = Field(default_factory=list, description="用户的目标、愿望或计划")


class EmotionalChatbot:
    """具备长期记忆能力的情感聊天小助手"""

    def __init__(self, api_key=None, memory_file="chat_memory.json", use_deepseek=True):
        """
        初始化情感聊天机器人

        Args:
            api_key: DeepSeek API密钥
            memory_file: 长期记忆存储文件路径
            use_deepseek: 是否使用DeepSeek模型（默认True）
        """
        self.use_deepseek = use_deepseek
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.memory_file = memory_file

        # 初始化LLM - 使用DeepSeek
        if self.use_deepseek:
            self.llm = ChatOpenAI(
                temperature=0.7,
                model="deepseek-chat",
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
            # 用于信息提取的LLM（温度设置更低以获得更准确的提取）
            self.extraction_llm = ChatOpenAI(
                temperature=0.1,
                model="deepseek-chat",
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        else:
            self.llm = ChatOpenAI(
                temperature=0.7,
                model="gpt-3.5-turbo",
                api_key=self.api_key
            )
            self.extraction_llm = ChatOpenAI(
                temperature=0.1,
                model="gpt-3.5-turbo",
                api_key=self.api_key
            )

        # 初始化短期记忆（对话历史）
        self.message_history = InMemoryChatMessageHistory()

        # 加载长期记忆
        self.long_term_memory = self._load_long_term_memory()

        # 创建信息提取的提示模板
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个信息提取专家。从用户的对话中提取以下类型的信息：

1. **用户信息**: 职业、居住地、年龄、家庭状况、教育背景等个人基本信息
2. **偏好**: 兴趣爱好、喜欢的活动、食物、音乐、书籍、电影等
3. **重要事件**: 生日、纪念日、重要约会、计划的活动等
4. **情感**: 当前的情绪状态、感受、心情等
5. **目标**: 想要达成的目标、愿望、计划等

请仔细分析用户的话，只提取明确提到的信息。如果某个类别没有相关信息，返回空列表。

以JSON格式返回，格式如下：
{{
  "user_info": ["信息1", "信息2"],
  "preferences": ["偏好1", "偏好2"],
  "important_events": ["事件1", "事件2"],
  "emotions": ["情感1", "情感2"],
  "goals": ["目标1", "目标2"]
}}"""),
            ("user", "用户说: {user_input}\n\n请提取其中的重要信息。")
        ])

        # 创建情感聊天提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个温暖、善解人意的情感陪伴助手。你的目标是：
1. 倾听用户的情感和想法
2. 提供情感支持和理解
3. 给予积极正面的回应
4. 记住用户分享的重要信息

用户背景信息：
{user_context}

请以温暖、共情的方式回应用户。如果用户分享了重要信息（如兴趣爱好、重要事件、个人偏好等），要记住并在未来的对话中体现出来。"""),
            ("human", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def _load_long_term_memory(self):
        """从文件加载长期记忆"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载记忆文件失败: {e}")
                return self._create_empty_memory()
        return self._create_empty_memory()

    def _create_empty_memory(self):
        """创建空的记忆结构"""
        return {
            "user_info": {},
            "important_events": [],
            "preferences": {},
            "emotions": {},
            "goals": {},
            "conversation_summary": [],
            "last_interaction": None
        }

    def _save_long_term_memory(self):
        """保存长期记忆到文件"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆文件失败: {e}")

    def _extract_important_info(self, user_input, ai_response):
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
                    self.long_term_memory["goals"][timestamp] = goal

            # 如果提取到了信息，打印日志（可选）
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
            print(f"[警告] JSON解析失败: {e}")
            print(f"原始响应: {extracted_text[:200]}")
        except Exception as e:
            print(f"[警告] 信息提取失败: {e}")

    def _get_user_context(self):
        """获取用户上下文信息"""
        context_parts = []

        if self.long_term_memory["user_info"]:
            recent_info = list(self.long_term_memory["user_info"].values())[-5:]
            context_parts.append("用户信息: " + "; ".join(recent_info))

        if self.long_term_memory["preferences"]:
            recent_prefs = list(self.long_term_memory["preferences"].values())[-5:]
            context_parts.append("用户偏好: " + "; ".join(recent_prefs))

        if self.long_term_memory["important_events"]:
            recent_events = self.long_term_memory["important_events"][-3:]
            events_str = "; ".join([e["content"] for e in recent_events])
            context_parts.append("重要事件: " + events_str)

        if self.long_term_memory["emotions"]:
            recent_emotions = list(self.long_term_memory["emotions"].values())[-3:]
            context_parts.append("最近情感: " + "; ".join(recent_emotions))

        if self.long_term_memory["goals"]:
            recent_goals = list(self.long_term_memory["goals"].values())[-3:]
            context_parts.append("用户目标: " + "; ".join(recent_goals))

        if self.long_term_memory["last_interaction"]:
            context_parts.append(f"上次交流: {self.long_term_memory['last_interaction']}")

        return "\n".join(context_parts) if context_parts else "这是你们的第一次对话"

    def chat(self, user_input):
        """
        与用户进行对话

        Args:
            user_input: 用户输入的消息

        Returns:
            助手的回复
        """
        # 获取用户上下文
        user_context = self._get_user_context()

        # 将历史消息添加到提示中
        messages = []
        for msg in self.message_history.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"type": "ai", "content": msg.content})

        # 生成回复
        response = self.chain.invoke({
            "input": user_input,
            "user_context": user_context
        })
        response_text = response.content if hasattr(response, 'content') else str(response)

        # 保存到短期记忆
        self.message_history.add_user_message(user_input)
        self.message_history.add_ai_message(response_text)

        # 提取并保存到长期记忆
        self._extract_important_info(user_input, response_text)

        self.long_term_memory["last_interaction"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._save_long_term_memory()

        return response_text

    def get_memory_summary(self):
        """获取记忆摘要"""
        return {
            "用户信息数量": len(self.long_term_memory["user_info"]),
            "偏好记录数量": len(self.long_term_memory["preferences"]),
            "重要事件数量": len(self.long_term_memory["important_events"]),
            "情感记录数量": len(self.long_term_memory["emotions"]),
            "目标数量": len(self.long_term_memory["goals"]),
            "最后交互时间": self.long_term_memory["last_interaction"]
        }

    def get_detailed_memory(self):
        """获取详细记忆内容"""
        return {
            "用户信息": list(self.long_term_memory["user_info"].values())[-5:],
            "偏好": list(self.long_term_memory["preferences"].values())[-5:],
            "重要事件": [e["content"] for e in self.long_term_memory["important_events"][-5:]],
            "情感": list(self.long_term_memory["emotions"].values())[-5:],
            "目标": list(self.long_term_memory["goals"].values())[-5:]
        }

    def clear_memory(self):
        """清除所有记忆"""
        self.message_history.clear()
        self.long_term_memory = self._create_empty_memory()
        self._save_long_term_memory()
        print("记忆已清除")


def main():
    """命令行交互模式"""
    print("=" * 50)
    print("情感聊天小助手 (DeepSeek V3 + 智能记忆)")
    print("=" * 50)
    print("我是你的情感陪伴助手，随时倾听你的心声")
    print("我会智能记住你分享的重要信息")
    print()
    print("可用命令:")
    print("  - 'quit' 或 'exit': 退出")
    print("  - 'memory': 查看记忆摘要")
    print("  - 'detail': 查看详细记忆")
    print("  - 'clear': 清除所有记忆")
    print("=" * 50)

    # 初始化聊天机器人
    chatbot = EmotionalChatbot()

    while True:
        try:
            user_input = input("\n你: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("再见！希望下次能再见到你~")
                break

            elif user_input.lower() == 'memory':
                summary = chatbot.get_memory_summary()
                print("\n记忆摘要:")
                for key, value in summary.items():
                    print(f"  {key}: {value}")
                continue

            elif user_input.lower() == 'detail':
                details = chatbot.get_detailed_memory()
                print("\n详细记忆:")
                for category, items in details.items():
                    print(f"\n  {category}:")
                    if items:
                        for i, item in enumerate(items, 1):
                            print(f"    {i}. {item}")
                    else:
                        print("    (暂无)")
                continue

            elif user_input.lower() == 'clear':
                chatbot.clear_memory()
                print("\n记忆已清除")
                continue

            elif not user_input:
                continue

            response = chatbot.chat(user_input)
            print(f"助手: {response}")

        except KeyboardInterrupt:
            print("\n\n再见！希望下次能再见到你~")
            break
        except Exception as e:
            print(f"\n抱歉，出现了一些问题: {e}")


if __name__ == "__main__":
    main()
