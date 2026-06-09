from fastapi import APIRouter, HTTPException
from datetime import datetime
import uuid
import logging

from api.schemas import CreateSessionRequest, SessionResponse
from api.exceptions import SessionNotFoundException
from db.crud import create_session, get_session

router = APIRouter()
logger = logging.getLogger(__name__)


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
