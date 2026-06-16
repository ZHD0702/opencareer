from fastapi import APIRouter, HTTPException
import logging

from api.schemas import EmotionTrendsResponse
from api.exceptions import SessionNotFoundException
from services.analysis_service import AnalysisService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/emotion/trends/{session_id}", response_model=EmotionTrendsResponse)
async def get_emotion_trends(session_id: str):
    """
    获取情绪趋势分析
    
    Args:
        session_id: 会话 ID
    
    Returns:
        情绪趋势分析数据
    """
    try:
        analysis_service = AnalysisService()
        
        trends = analysis_service.get_emotion_trends(session_id)
        if not trends:
            raise SessionNotFoundException(session_id)
        
        logger.info(f"获取情绪趋势: session_id={session_id}")
        
        return EmotionTrendsResponse(**trends)
    except SessionNotFoundException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取情绪趋势失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取情绪趋势失败，请稍后重试")
