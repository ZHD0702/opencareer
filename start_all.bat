@echo off
chcp 65001 >nul
echo ========================================
echo OpenCareer - 快速启动脚本
echo ========================================
echo.

cd /d %~dp0

echo [1/3] 启动 MCP 服务器...
start "OpenCareer MCP Server" cmd /k "python -m opencareer.mcp.server"

timeout /t 2 /nobreak >nul

echo.
echo [2/3] 启动后端服务器...
start "OpenCareer Backend" cmd /k "cd web\backend && python main.py"

timeout /t 2 /nobreak >nul

echo.
echo [3/3] 启动前端服务器...
cd web\front
if not exist "node_modules" (
    echo 首次运行，安装 npm 依赖...
    call npm install
)
start "OpenCareer Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo 所有服务已启动！
echo - 后端: http://localhost:8000
echo - 前端: http://localhost:5173
echo - MCP: http://localhost:8001
echo.
echo 使用说明:
echo - 后端已集成 OpenCareer 完整的 Agent 架构
echo - 默认使用外面的 opencareer 包中的 CareerAgent
echo - 如需启用 MCP 工具，请设置 CAREER_USE_MCP=true
echo ========================================
echo.
echo 按任意键退出此窗口（服务将继续运行）...
pause >nul
