from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel, Field
import logging

from api.exceptions import SessionNotFoundException
from services.analysis_service import AnalysisService

router = APIRouter()
logger = logging.getLogger(__name__)


class SkillAssessmentUpdate(BaseModel):
    target_role: Optional[str] = None
    skills: list[dict] = Field(default_factory=list)


@router.get("/skill-assessment/{session_id}")
async def get_skill_assessment(session_id: str):
    """
    获取技能评估
    
    Args:
        session_id: 会话 ID
    
    Returns:
        技能评估数据
    """
    try:
        analysis_service = AnalysisService()
        assessment = analysis_service.get_skill_assessment(session_id)
        
        if not assessment:
            raise SessionNotFoundException(session_id)
        
        logger.info(f"获取技能评估: session_id={session_id}")
        
        return assessment
    except SessionNotFoundException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取技能评估失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取技能评估失败，请稍后重试")


@router.patch("/skill-assessment/{session_id}")
async def update_skill_assessment(session_id: str, payload: SkillAssessmentUpdate):
    """
    更新技能评估
    
    Args:
        session_id: 会话 ID
        target_role: 目标角色
        skills: 技能列表
    
    Returns:
        更新后的技能评估
    """
    try:
        analysis_service = AnalysisService()
        assessment = analysis_service.update_skill_assessment(session_id, payload.target_role, payload.skills)
        
        if not assessment:
            raise SessionNotFoundException(session_id)
        
        logger.info(f"更新技能评估: session_id={session_id}")
        
        return assessment
    except SessionNotFoundException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新技能评估失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="更新技能评估失败，请稍后重试")
