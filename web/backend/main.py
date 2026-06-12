from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback

from api.routes import chat, sessions, emotion, skill, resume, mcp, progress
from api.routes import careers, orchestrator
from db.crud import init_sync_db
from careers_config import config
from adapters.mcp_service import start_mcp_server, stop_mcp_server, get_mcp_service

# 先加载项目根目录的 .env
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent  # F:\opencareer\web\backend\main.py -> F:\opencareer
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
# 然后加载本地 .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("OpenCareer API 启动中...")
    logger.info("=" * 60)
    init_sync_db()
    logger.info("✅ 数据库初始化完成")
    
    # 启动 MCP 服务器（如果启用）
    if config.USE_MCP:
        logger.info(f"📦 正在启动 MCP 服务器 (URL: {config.MCP_URL})...")
        mcp_started = await start_mcp_server()
        if mcp_started:
            logger.info("✅ MCP 服务器启动成功 - resume-skill 已就绪")
        else:
            logger.warning("⚠️ MCP 服务器启动失败 - 将使用无工具模式运行")
    else:
        logger.info("ℹ️ MCP 功能已禁用 (CAREER_USE_MCP=false)")
    
    yield
    
    logger.info("=" * 60)
    logger.info("OpenCareer API 关闭中...")
    logger.info("=" * 60)
    
    # 关闭 MCP 服务器
    if config.USE_MCP:
        await stop_mcp_server()
        logger.info("✅ MCP 服务器已关闭")


app = FastAPI(
    title="OpenCareer API",
    description="Backend for OpenCareer - Frontend Talker, Backend Thinker",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 异常处理器
@app.exception_handler(StarletteHTTPException)
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


@app.exception_handler(RequestValidationError)
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


@app.exception_handler(Exception)
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


# 注册路由
app.include_router(sessions.router, prefix="/api", tags=["sessions"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(careers.router, prefix="/api", tags=["careers"])
app.include_router(emotion.router, prefix="/api", tags=["emotion"])
app.include_router(skill.router, prefix="/api", tags=["skill"])
app.include_router(progress.router, prefix="/api", tags=["job-progress"])
app.include_router(resume.router, prefix="/api", tags=["resume"])
app.include_router(mcp.router, prefix="/api", tags=["mcp"])
app.include_router(orchestrator.router, prefix="/api", tags=["orchestrator", "llm", "knowledge"])


@app.get("/")
async def root():
    return {"status": "ok", "message": "OpenCareer API"}


@app.get("/health")
async def health():
    """健康检查端点 - 包含 MCP 服务状态"""
    mcp_service = await get_mcp_service()
    mcp_status = {
        "mcp_enabled": config.USE_MCP,
        "mcp_running": mcp_service.is_running() if mcp_service else False,
        "mcp_port": config.MCP_PORT if config.USE_MCP else None,
    }
    return {
        "status": "healthy",
        "mcp": mcp_status if config.USE_MCP else {"enabled": False}
    }


@app.middleware("http")
async def log_requests(request, call_next):
    import time
    start_time = time.time()
    
    logger.info(f"请求开始: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"请求完成: {request.method} {request.url.path} - "
        f"状态码: {response.status_code} - "
        f"耗时: {process_time:.3f}秒"
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False
    )
