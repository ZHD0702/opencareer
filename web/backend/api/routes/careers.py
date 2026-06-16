from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import logging
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Any

from api.exceptions import SessionNotFoundException, LLMServiceException
from db.crud import save_message, get_messages, get_session
from agent_factory import get_agent_factory
from careers_config import config

router = APIRouter()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
JOB_DATA_PATH = PROJECT_ROOT / "岗位匹配数据.xlsx"
MATCH_TEMP_DIR = Path(__file__).resolve().parents[2] / "data" / "match"


class CareerMatchRequest(BaseModel):
    mode: str = "auto"
    top_n: int = 10
    coarse_n: int = 30
    final_n: int = 5


def _normalize_memory_for_matcher(memory: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(memory)
    goals = memory.get("goals", [])
    if isinstance(goals, list):
        normalized["goals"] = {}
        for item in goals:
            if isinstance(item, dict):
                ts = item.get("timestamp", "")
                normalized["goals"][ts] = item.get("content", str(item))
            else:
                normalized["goals"][str(len(normalized["goals"]))] = str(item)
    return normalized


def _dataframe_records(df) -> list[dict[str, Any]]:
    records = []
    for row in df.to_dict(orient="records"):
        cleaned = {}
        for key, value in row.items():
            if hasattr(value, "item"):
                value = value.item()
            if value != value:
                value = None
            cleaned[str(key)] = value
        records.append(cleaned)
    return records


def _run_local_match(memory_path: Path, top_n: int) -> list[dict[str, Any]]:
    from career_matcher import match_career

    result = match_career(
        excel_path=str(JOB_DATA_PATH),
        memory_path=str(memory_path),
        top_n=top_n,
    )
    return _dataframe_records(result)


def _run_ai_match(memory_path: Path, coarse_n: int, final_n: int) -> str:
    from career_matcher_ai import ai_match_career

    return ai_match_career(
        excel_path=str(JOB_DATA_PATH),
        memory_path=str(memory_path),
        coarse_n=coarse_n,
        final_n=final_n,
    )

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
            memory_file=str(config.get_session_memory_path(session_id))
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


@router.post("/careers/match/{session_id}")
async def match_careers(session_id: str, payload: CareerMatchRequest):
    """Run the CLI job matcher through the GUI backend."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    if not JOB_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Job data file not found: {JOB_DATA_PATH}")

    agent = await _get_career_agent(session_id)
    memory = getattr(agent, "long_term_memory", {}) or {}
    if not memory.get("user_info") and not memory.get("goals") and not memory.get("preferences"):
        raise HTTPException(status_code=400, detail="No user profile found. Chat with the assistant first.")

    MATCH_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    memory_path = MATCH_TEMP_DIR / f"{session_id}.json"
    memory_path.write_text(
        json.dumps(_normalize_memory_for_matcher(memory), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mode = payload.mode.lower()
    try:
        if mode in {"auto", "ai"}:
            try:
                report = await asyncio.to_thread(
                    _run_ai_match,
                    memory_path,
                    payload.coarse_n,
                    payload.final_n,
                )
                return {
                    "success": True,
                    "mode": "ai",
                    "report": report,
                    "recommendations": [],
                    "fallback": False,
                }
            except Exception as exc:
                if mode == "ai":
                    raise
                logger.warning("AI career match failed, falling back to local matcher: %s", exc)

        recommendations = await asyncio.to_thread(_run_local_match, memory_path, payload.top_n)
        return {
            "success": True,
            "mode": "local",
            "report": "",
            "recommendations": recommendations,
            "fallback": mode == "auto",
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Career match failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Career match failed: {exc}") from exc
    finally:
        try:
            os.remove(memory_path)
        except OSError:
            pass


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
