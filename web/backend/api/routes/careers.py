from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import json
import logging
from typing import Optional

from api.exceptions import SessionNotFoundException, LLMServiceException
from db.crud import save_message, get_messages, get_session
from agent_factory import get_agent_factory
from careers_config import config

router = APIRouter()
logger = logging.getLogger(__name__)

# 全局 CareerAgent 实例管理（每个会话一个实例，或单例）
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


@router.post("/careers/chat/{session_id}")
async def careers_chat_stream(session_id: str, request: Request):
    """
    Career Agent SSE 流式对话接口 (Phase 3)
    
    使用完整的 CareerAgent，支持 MCP 工具调用、记忆管理等功能
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
        
        # 获取 CareerAgent 实例
        agent = await _get_career_agent(session_id)
        
        logger.info("开始调用 CareerAgent 生成回复")
        
        # 收集完整的 AI 回复用于保存到数据库
        full_ai_response = []
        
        async def event_generator():
            try:
                # 检查是 CareerAgent 还是 SimpleAgent
                if hasattr(agent, "stream_chat"):
                    # CareerAgent - 使用流式聊天
                    async for token in agent.stream_chat(user_message):
                        full_ai_response.append(token)
                        yield f"data: {json.dumps({'type': 'sentence', 'content': token})}\n\n"
                else:
                    # SimpleAgent 回退
                    if hasattr(agent, "chat"):
                        # 非流式，但也包装成流式返回
                        response = await agent.chat(user_message) if hasattr(agent.chat, "__await__") else agent.chat(user_message)
                        full_ai_response.append(response)
                        yield f"data: {json.dumps({'type': 'sentence', 'content': response})}\n\n"
                
                # 所有事件发送完毕后，保存完整回复到数据库
                if full_ai_response:
                    full_text = "".join(full_ai_response)
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
        "errors": errors,
        "phase": "phase_3",
        "status": "career_agent_integrated"
    }


@router.get("/careers/memory/{session_id}")
async def get_agent_memory(session_id: str):
    """获取 Agent 记忆摘要"""
    try:
        agent = await _get_career_agent(session_id)
        
        if hasattr(agent, "get_memory_summary"):
            memory_summary = agent.get_memory_summary()
            return {
                "success": True,
                "memory": memory_summary,
                "phase": "phase_3"
            }
        else:
            return {
                "success": True,
                "memory": None,
                "message": "当前使用的 Agent 不支持记忆功能",
                "phase": "phase_3"
            }
    except Exception as e:
        logger.error(f"获取记忆失败: {e}", exc_info=True)
        return {
            "success": False,
            "memory": None,
            "message": str(e),
            "phase": "phase_3"
        }


@router.delete("/careers/memory/{session_id}")
async def clear_agent_memory(session_id: str):
    """清除 Agent 记忆"""
    try:
        agent = await _get_career_agent(session_id)
        
        if hasattr(agent, "clear_memory"):
            agent.clear_memory()
            return {
                "success": True,
                "message": "记忆已清除",
                "phase": "phase_3"
            }
        else:
            return {
                "success": False,
                "message": "当前使用的 Agent 不支持记忆清除功能",
                "phase": "phase_3"
            }
    except Exception as e:
        logger.error(f"清除记忆失败: {e}", exc_info=True)
        return {
            "success": False,
            "message": str(e),
            "phase": "phase_3"
        }
