from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
import logging

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP异常处理器"""
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail} - {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "type": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理器"""
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(f"验证异常: {errors} - {request.url.path}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "type": "validation_error",
            "message": "请求数据验证失败",
            "details": errors
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logger.error(
        f"未处理的异常: {str(exc)}\n"
        f"路径: {request.url.path}\n"
        f"方法: {request.method}\n"
        f"堆栈: {traceback.format_exc()}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "type": "internal_error",
            "message": "服务器内部错误，请稍后重试",
            "status_code": 500
        }
    )
