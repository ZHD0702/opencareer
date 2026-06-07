@echo off
chcp 65001 >nul
echo ============================================================
echo OpenCareer Web 后端启动脚本
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/3] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 Python，请先安装 Python
    pause
    exit /b 1
)
echo ✅ Python 已就绪

echo.
echo [2/3] 检查依赖...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo ⚠️  正在安装依赖...
    pip install -r requirements.txt -q
)
echo ✅ 依赖已就绪

echo.
echo [3/3] 启动后端服务...
echo ============================================================
echo 启动后端 (端口: 8080)
echo MCP 功能: 已启用
echo ============================================================
echo.
echo 启动后访问: http://localhost:8080
echo 健康检查:  http://localhost:8080/health
echo.
echo 按 Ctrl+C 停止服务
echo ============================================================
echo.

cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8080

pause
