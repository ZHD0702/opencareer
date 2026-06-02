from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import logging
from typing import Optional

from api.exceptions import SessionNotFoundException, LLMServiceException
from db.crud import save_message, get_messages, get_session
from agent_factory import get_agent_factory, AgentType
from agent_adapter import create_adapter
from careers_config import config

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    agent_type: Optional[str] = None


@router.post("/careers/chat/{session_id}")
async def careers_chat_stream(session_id: str, request: Request):
    """
    Career Agent SSE 流式对话接口
    
    使用 CareerAgent (LangChain + MCP) 处理对话
    """
    try:
        logger.info(f"收到 Career Agent 聊天请求: session_id={session_id}")
        
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
        
        logger.info("保存用户消息到数据库")
        save_message(session_id, "user", user_message.strip())
        
        messages = get_messages(session_id, limit=50)
        logger.info(f"获取到 {len(messages)} 条历史消息")
        
        # 构建历史消息
        history = [
            {"role": "assistant" if msg["role"] == "ai" else msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] in ["user", "ai"]
        ]
        
        logger.info(f"构建了 {len(history)} 条上下文用于 CareerAgent")
        
        # 使用 Agent 工厂获取 CareerAgent
        try:
            factory = get_agent_factory()
            agent = factory.create_agent(AgentType.CAREER)
            agent = create_adapter(agent)
            logger.info("CareerAgent 初始化成功")
        except Exception as e:
            logger.error(f"CareerAgent 初始化失败: {str(e)}", exc_info=True)
            # 回退到 SimpleAgent
            logger.warning("回退到 SimpleAgent")
            factory = get_agent_factory()
            agent = factory.create_agent(AgentType.SIMPLE)
        
        logger.info("开始调用 CareerAgent 生成回复")
        
        # 收集完整的 AI 回复用于保存到数据库
        full_ai_response = []
        
        async def event_generator():
            try:
                # 让 Agent 处理并流式返回
                async for event in agent.chat(session_id, history):
                    # 解析事件，收集 AI 回复
                    if event.startswith('data: '):
                        try:
                            data_str = event[6:].strip()
                            if data_str:
                                data = json.loads(data_str)
                                if data.get('type') == 'sentence':
                                    full_ai_response.append(data.get('content', ''))
                        except:
                            pass
                    
                    yield event
                
                # 所有事件发送完毕后，保存完整回复到数据库
                if full_ai_response:
                    full_text = "".join(full_ai_response)
                    logger.info(f"保存 AI 回复到数据库: {len(full_text)} 字符")
                    save_message(session_id, "ai", full_text)
                
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
        logger.error(f"Career Agent 聊天接口错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误，请稍后重试")


@router.get("/careers/config")
async def get_careers_config():
    """获取 Career Agent 配置信息"""
    is_valid, errors = config.validate_config()
    return {
        "agent_type": config.AGENT_TYPE,
        "use_mcp": config.USE_MCP,
        "mcp_url": config.MCP_URL,
        "is_valid": is_valid,
        "errors": errors
    }


@router.get("/careers/memory/{session_id}")
async def get_agent_memory(session_id: str):
    """获取 Agent 记忆摘要（如果有）"""
    try:
        factory = get_agent_factory()
        agent = factory.get_or_create_agent(session_id, AgentType.CAREER)
        
        if hasattr(agent, 'get_memory_summary'):
            return {
                "success": True,
                "memory": agent.get_memory_summary()
            }
        else:
            return {
                "success": True,
                "memory": None,
                "message": "当前 Agent 不支持记忆功能"
            }
    except Exception as e:
        logger.error(f"获取记忆失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


@router.delete("/careers/memory/{session_id}")
async def clear_agent_memory(session_id: str):
    """清除 Agent 记忆"""
    try:
        factory = get_agent_factory()
        agent = factory.get_or_create_agent(session_id, AgentType.CAREER)
        
        if hasattr(agent, 'clear_memory'):
            agent.clear_memory()
            factory.clear_agent_cache(session_id)
            return {
                "success": True,
                "message": "记忆已清除"
            }
        else:
            factory.clear_agent_cache(session_id)
            return {
                "success": True,
                "message": "Agent 缓存已清除"
            }
    except Exception as e:
        logger.error(f"清除记忆失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }
