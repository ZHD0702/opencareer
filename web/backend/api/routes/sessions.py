from fastapi import APIRouter, HTTPException
from datetime import datetime
import uuid
import logging
import re

from api.schemas import CreateSessionRequest, SessionResponse
from api.exceptions import SessionNotFoundException
from db.crud import create_session, delete_session, get_messages, get_session, list_sessions
from services.job_search_readiness import build_job_match_action

router = APIRouter()
logger = logging.getLogger(__name__)


def _clean_session_preview(content: str) -> str:
    return re.sub(r"\s*\[调用工具\s*[:：][^\]]+\]\s*", " ", content or "").strip()


def _restore_latest_job_action(messages: list[dict], action: dict | None) -> list[dict]:
    restored = [dict(message) for message in messages]
    if not action or not restored or restored[-1].get("role") != "ai":
        return restored
    latest = restored[-1]
    actions = list(latest.get("actions") or [])
    if not any(item.get("action") == action["action"] for item in actions):
        latest["actions"] = [*actions, action]
    return restored


@router.get("/sessions")
async def list_user_sessions(user_id: str = "default_user", limit: int = 100):
    """
    获取历史会话列表。
    """
    try:
        sessions = list_sessions(user_id=user_id, limit=limit)
        result = []

        for session in sessions:
            messages = get_messages(session["session_id"], limit=1)
            all_messages = get_messages(session["session_id"], limit=500)
            user_messages = [m for m in all_messages if m["role"] == "user"]
            if not user_messages:
                continue

            latest = messages[0] if messages else None
            turn_count = len([m for m in all_messages if m["role"] in {"user", "ai"}])
            fallback_title = "新的会话"
            first_user = user_messages[-1]
            fallback_title = first_user["content"][:14]

            result.append({
                "session_id": session["session_id"],
                "user_id": session["user_id"],
                "title": session.get("title") or fallback_title,
                "preview": _clean_session_preview(latest["content"])[:80] if latest else "",
                "turn_count": turn_count,
                "current_agent": session.get("current_phase"),
                "created_at": session.get("updated_at") or session.get("created_at"),
            })

        return result
    except Exception as e:
        logger.error(f"获取历史会话列表失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取历史会话列表失败，请稍后重试")


@router.post("/sessions", response_model=SessionResponse)
async def create_new_session(request: CreateSessionRequest):
    """
    创建新会话
    
    Args:
        request: 创建会话请求
    
    Returns:
        新创建的会话信息
    """
    try:
        if not request.user_id or not request.user_id.strip():
            raise HTTPException(status_code=400, detail="用户ID不能为空")
        
        session_id = str(uuid.uuid4())
        
        session = create_session(
            session_id=session_id,
            user_id=request.user_id.strip(),
            target_role=request.target_role.strip() if request.target_role else None
        )
        
        logger.info(f"创建新会话: {session_id}, 用户: {request.user_id}")
        
        return SessionResponse(
            session_id=session["session_id"],
            user_id=session["user_id"],
            target_role=session["target_role"],
            created_at=session["created_at"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建会话失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="创建会话失败，请稍后重试")


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session_info(session_id: str):
    """
    获取会话信息
    
    Args:
        session_id: 会话ID
    
    Returns:
        会话信息
    """
    try:
        session = get_session(session_id)
        if not session:
            raise SessionNotFoundException(session_id)
        
        return SessionResponse(
            session_id=session["session_id"],
            user_id=session["user_id"],
            target_role=session["target_role"],
            created_at=session["created_at"]
        )
    except SessionNotFoundException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话信息失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取会话信息失败，请稍后重试")


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 200):
    """
    获取会话历史消息，用于页面重新打开时恢复聊天记录。
    """
    try:
        session = get_session(session_id)
        if not session:
            raise SessionNotFoundException(session_id)

        safe_limit = max(1, min(limit, 500))
        messages = get_messages(session_id, limit=safe_limit)
        messages = list(reversed(messages))
        messages = _restore_latest_job_action(
            messages,
            build_job_match_action(session_id),
        )

        return {
            "session_id": session_id,
            "messages": [
                {
                    "id": str(message["id"]),
                    "role": "ai" if message["role"] == "ai" else "user",
                    "content": message["content"],
                    "actions": message.get("actions") or [],
                    "created_at": message["created_at"],
                }
                for message in messages
                if message["role"] in {"user", "ai"}
            ],
        }
    except SessionNotFoundException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话历史消息失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取会话历史消息失败，请稍后重试")


@router.delete("/sessions/{session_id}")
async def delete_user_session(session_id: str):
    """
    删除历史会话。
    """
    try:
        deleted = delete_session(session_id)
        return {"deleted": deleted}
    except Exception as e:
        logger.error(f"删除会话失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="删除会话失败，请稍后重试")
