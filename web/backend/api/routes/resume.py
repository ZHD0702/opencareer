from fastapi import APIRouter, HTTPException
import logging

from api.schemas import ResumeResponse, ResumeData
from api.exceptions import SessionNotFoundException
from services.analysis_service import AnalysisService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/resume/{session_id}", response_model=ResumeResponse)
async def get_resume(session_id: str):
    """
    获取简历信息
    
    Args:
        session_id: 会话 ID
    
    Returns:
        简历信息数据
    """
    try:
        analysis_service = AnalysisService()
        resume_data = analysis_service.get_resume(session_id)
        
        if not resume_data:
            raise SessionNotFoundException(session_id)
        
        logger.info(f"获取简历信息: session_id={session_id}")
        
        return ResumeResponse(**resume_data)
    except SessionNotFoundException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取简历信息失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取简历信息失败，请稍后重试")


@router.patch("/resume/{session_id}", response_model=ResumeResponse)
async def update_resume(session_id: str, data: ResumeData):
    """
    更新简历信息
    
    Args:
        session_id: 会话 ID
        data: 简历数据
    
    Returns:
        更新后的简历信息
    """
    try:
        analysis_service = AnalysisService()
        resume_data = analysis_service.update_resume(session_id, data.model_dump())
        
        if not resume_data:
            raise SessionNotFoundException(session_id)
        
        logger.info(f"更新简历信息: session_id={session_id}")
        
        return ResumeResponse(**resume_data)
    except SessionNotFoundException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新简历信息失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="更新简历信息失败，请稍后重试")
