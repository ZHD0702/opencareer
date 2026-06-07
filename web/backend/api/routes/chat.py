from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import json
import logging
import asyncio

from api.exceptions import SessionNotFoundException, LLMServiceException
from db.crud import save_message, get_messages, get_session
from agent_factory import get_agent_factory
from careers_config import config
from utils.text_fragmenter import TextFragmenter

router = APIRouter()
logger = logging.getLogger(__name__)

# 全局 CareerAgent 实例管理（每个会话一个实例）
_career_agents = {}


async def _get_career_agent(session_id: str):
    """获取或创建 CareerAgent 实例"""
    if session_id not in _career_agents:
        factory = get_agent_factory()
        agent = factory.create_agent(
            "career",
            mcp_url=config.MCP_URL,
            use_mcp=config.USE_MCP,
            memory_file=f"career_memory_{session_id}.json"
        )
        
        # 连接 MCP（如果启用）
        if hasattr(agent, "connect_mcp") and config.USE_MCP:
            try:
                await agent.connect_mcp()
            except Exception as e:
                logger.warning(f"Failed to connect MCP for session {session_id}: {e}")
        
        _career_agents[session_id] = agent
    
    return _career_agents[session_id]


@router.post("/chat/{session_id}")
async def chat_stream(session_id: str, request: Request):
    """
    SSE 流式对话接口 - 微信风格碎片化聊天
    
    优先使用 CareerAgent（Phase 3），带有完整的 LangChain 记忆系统
    支持智能断句，自动创建多个气泡
    """
    try:
        logger.info(f"收到聊天请求: session_id={session_id}")
        
        body = await request.json()
        user_message = body.get("message", "")
        
        logger.info(f"用户消息: \"{user_message}\"")
        
        if not user_message or not user_message.strip():
            logger.warning("消息内容为空")
            raise HTTPException(status_code=400, detail="消息内容不能为空")
        
        if len(user_message) > 2000:
            logger.warning(f"消息过长: {len(user_message)} 字符")
            raise HTTPException(status_code=400, detail="消息内容过长，请控制在2000字以内")
        
        session = get_session(session_id)
        if not session:
            logger.warning(f"会话不存在: {session_id}")
            raise SessionNotFoundException(session_id)
        
        logger.info(f"保存用户消息到数据库")
        save_message(session_id, "user", user_message.strip())
        
        # 获取 CareerAgent 实例（优先使用）
        agent = await _get_career_agent(session_id)
        
        logger.info("开始调用 Agent 生成回复")
        
        # 收集完整的 AI 回复用于保存到数据库
        full_ai_response = []
        
        async def event_generator():
            try:
                # 先发送"对方正在输入..."信号
                yield f"data: {json.dumps({'type': 'typing_start'})}\n\n"
                
                # 检查是否是 CareerAgent
                if hasattr(agent, "stream_chat"):
                    # CareerAgent - 使用完整的 LangChain 记忆系统
                    logger.info("使用 CareerAgent (带有 LangChain 记忆系统)")
                    
                    async for token in agent.stream_chat(user_message):
                        # 检查是否是断句信号（跳过）
                        if token == "__FRAGMENT_BREAK__":
                            continue
                        else:
                            # 只收集内容，不发送流式输出
                            full_ai_response.append(token)
                    
                else:
                    # SimpleAgent 回退
                    logger.info("使用 SimpleAgent (回退)")
                    
                    messages = get_messages(session_id, limit=50)
                    history = [
                        {"role": "assistant" if msg["role"] == "ai" else msg["role"], "content": msg["content"]}
                        for msg in messages
                        if msg["role"] in ["user", "ai"]
                    ]
                    
                    if hasattr(agent, "chat"):
                        async for event in agent.chat(session_id, history):
                            if event.startswith('data: '):
                                try:
                                    data_str = event[6:].strip()
                                    if data_str:
                                        data = json.loads(data_str)
                                        if data.get('type') == 'sentence':
                                            full_ai_response.append(data.get('content', ''))
                                except:
                                    pass
                            # 不发送事件，只收集内容
                
                # 发送"对方正在输入结束"信号
                yield f"data: {json.dumps({'type': 'typing_end'})}\n\n"
                
                # 对回复进行碎片化处理
                if full_ai_response:
                    full_text = "".join(full_ai_response)
                    logger.info(f"AI回复完成: {len(full_text)} 字符，开始碎片化处理")
                    
                    # 使用碎片化器处理文本（每次创建新实例确保加载最新代码）
                    fragmenter = TextFragmenter()
                    fragments = fragmenter.fragment_with_particles(full_text)
                    
                    logger.info(f"碎片化完成: 生成 {len(fragments)} 个碎片")
                    
                    # 发送碎片化信号
                    yield f"data: {json.dumps({'type': 'fragmentation_start', 'count': len(fragments)})}\n\n"
                    
                    # 逐个发送碎片（带延迟）
                    for i, fragment in enumerate(fragments):
                        # 发送碎片
                        yield f"data: {json.dumps({
                            'type': 'fragment',
                            'content': fragment.content,
                            'emotion': fragment.emotion,
                            'index': i
                        })}\n\n"
                        
                        # 碎片之间添加延迟（最后一个碎片不需要延迟）
                        if i < len(fragments) - 1:
                            await asyncio.sleep(fragment.delay_ms / 1000.0)
                    
                    # 发送碎片化结束信号
                    yield f"data: {json.dumps({'type': 'fragmentation_end'})}\n\n"
                    
                    # 保存原始回复到数据库（不含语气词）
                    logger.info(f"保存 AI 回复到数据库: {len(full_text)} 字符")
                    save_message(session_id, "ai", full_text)
                
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                logger.error(f"事件生成器错误: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': '处理请求时发生错误'})}\n\n"
        
        logger.info("返回 StreamingResponse")
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except SessionNotFoundException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except LLMServiceException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"聊天接口错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误，请稍后重试")


@router.get("/chat/memory/{session_id}")
async def get_chat_memory(session_id: str):
    """获取对话记忆摘要（兼容原接口）"""
    try:
        agent = await _get_career_agent(session_id)
        
        if hasattr(agent, "get_memory_summary"):
            memory_summary = agent.get_memory_summary()
            return {
                "success": True,
                "memory": memory_summary,
                "agent_type": "career_agent",
                "has_langchain_memory": True
            }
        else:
            return {
                "success": True,
                "memory": None,
                "agent_type": "simple_agent",
                "has_langchain_memory": False,
                "message": "当前使用的 Agent 不支持 LangChain 记忆功能"
            }
    except Exception as e:
        logger.error(f"获取记忆失败: {e}", exc_info=True)
        return {
            "success": False,
            "memory": None,
            "message": str(e)
        }


@router.delete("/chat/memory/{session_id}")
async def clear_chat_memory(session_id: str):
    """清除对话记忆（兼容原接口）"""
    try:
        agent = await _get_career_agent(session_id)
        
        if hasattr(agent, "clear_memory"):
            agent.clear_memory()
            return {
                "success": True,
                "message": "记忆已清除",
                "agent_type": "career_agent"
            }
        else:
            return {
                "success": False,
                "message": "当前使用的 Agent 不支持记忆清除功能",
                "agent_type": "simple_agent"
            }
    except Exception as e:
        logger.error(f"清除记忆失败: {e}", exc_info=True)
        return {
            "success": False,
            "message": str(e)
        }
