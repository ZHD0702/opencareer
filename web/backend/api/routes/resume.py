from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import logging
from urllib.parse import quote

from api.schemas import ResumeResponse, ResumeData
from api.exceptions import SessionNotFoundException
from services.analysis_service import AnalysisService
from services.resume_pdf_service import (
    get_session_resume_pdf,
    get_session_resume_pdf_by_id,
    list_session_resume_pdfs,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _resume_pdf_payload(session_id: str, document: dict) -> dict:
    document_id = document["id"]
    version = document.get("created_at") or ""
    return {
        "id": document_id,
        "session_id": session_id,
        "filename": document["file_name"],
        "created_at": document.get("created_at"),
        "preview_url": f"/api/resume/{session_id}/pdfs/{document_id}/content?v={version}",
        "download_url": f"/api/resume/{session_id}/pdfs/{document_id}/download",
    }


@router.get("/resume/{session_id}/pdfs")
async def list_resume_pdfs(session_id: str):
    return [
        _resume_pdf_payload(session_id, document)
        for document in list_session_resume_pdfs(session_id)
    ]


@router.get("/resume/{session_id}/pdfs/{document_id}/content")
async def preview_resume_pdf_version(session_id: str, document_id: int):
    document = get_session_resume_pdf_by_id(session_id, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="未找到该 PDF 简历版本")
    return _pdf_response(document, inline=True)


@router.get("/resume/{session_id}/pdfs/{document_id}/download")
async def download_resume_pdf_version(session_id: str, document_id: int):
    document = get_session_resume_pdf_by_id(session_id, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="未找到该 PDF 简历版本")
    return _pdf_response(document, inline=False)


def _pdf_response(document: dict, inline: bool) -> FileResponse:
    if not inline:
        return FileResponse(
            path=document["path"],
            media_type="application/pdf",
            filename=document["file_name"],
        )
    return FileResponse(
        path=document["path"],
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                "inline; filename=resume.pdf; "
                f"filename*=UTF-8''{quote(document['file_name'])}"
            )
        },
    )


@router.get("/resume/{session_id}/pdf")
async def get_resume_pdf_info(session_id: str):
    document = get_session_resume_pdf(session_id)
    if not document:
        raise HTTPException(status_code=404, detail="该会话尚未生成 PDF 简历")
    return _resume_pdf_payload(session_id, document)


@router.get("/resume/{session_id}/pdf/content")
async def preview_resume_pdf(session_id: str):
    document = get_session_resume_pdf(session_id)
    if not document:
        raise HTTPException(status_code=404, detail="该会话尚未生成 PDF 简历")
    return _pdf_response(document, inline=True)


@router.get("/resume/{session_id}/pdf/download")
async def download_resume_pdf(session_id: str):
    document = get_session_resume_pdf(session_id)
    if not document:
        raise HTTPException(status_code=404, detail="该会话尚未生成 PDF 简历")
    return _pdf_response(document, inline=False)


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
