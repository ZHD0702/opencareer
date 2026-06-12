@echo off
setlocal EnableExtensions
chcp 65001 >nul
title OpenCareer 启动器

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%web\backend"
set "FRONTEND_DIR=%ROOT%web\front"
set "FRONTEND_URL=http://localhost:5173"

if /I "%~1"=="--check" goto CHECK

echo ========================================
echo OpenCareer 一键启动
echo ========================================
echo.

call :VALIDATE
if errorlevel 1 goto FAILED

echo [1/4] 检查后端和 MCP 依赖...
python -c "import fastapi, uvicorn, mcp" >nul 2>nul
if errorlevel 1 (
    echo 正在安装后端和 MCP 依赖，请稍等...
    python -m pip install -r "%BACKEND_DIR%\requirements.txt"
    if errorlevel 1 goto FAILED
)

echo.
echo [2/4] 检查前端依赖...
if not exist "%FRONTEND_DIR%\node_modules" (
    echo 正在安装前端依赖，请稍等...
    pushd "%FRONTEND_DIR%"
    call npm.cmd install
    if errorlevel 1 (
        popd
        goto FAILED
    )
    popd
)

echo.
echo [3/4] 启动后端和 MCP...
echo 后端地址: http://localhost:8000
echo MCP 地址:  http://localhost:8001/mcp
start "OpenCareer Backend + MCP" /D "%BACKEND_DIR%" cmd /k "set CAREER_USE_MCP=true&&set HOST=127.0.0.1&&set PORT=8000&&python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

echo 正在等待后端和 MCP 初始化完成...
call :WAIT_FOR_BACKEND
if errorlevel 1 (
    echo 后端暂未就绪，前端仍会启动并自动重试连接。
)

echo.
echo [4/4] 启动前端...
echo 前端地址: %FRONTEND_URL%
start "OpenCareer Frontend" /D "%FRONTEND_DIR%" cmd /k "npm.cmd run dev -- --host 127.0.0.1 --port 5173"

timeout /t 3 /nobreak >nul
start "" "%FRONTEND_URL%"

echo.
echo ========================================
echo OpenCareer 已启动
echo ========================================
echo 浏览器地址: %FRONTEND_URL%
echo.
echo 后端窗口会托管 MCP: 后端启动时自动检查并启动 8001 端口的 MCP 服务。
echo 关闭服务时，请分别在后端和前端窗口按 Ctrl+C。
echo.
pause
exit /b 0

:CHECK
call :VALIDATE
if errorlevel 1 exit /b 1
echo OpenCareer 启动脚本检查通过。
echo 根目录: %ROOT%
echo 后端: %BACKEND_DIR%
echo 前端: %FRONTEND_DIR%
echo 前端访问地址: %FRONTEND_URL%
exit /b 0

:VALIDATE
if not exist "%BACKEND_DIR%\main.py" (
    echo 未找到后端入口: "%BACKEND_DIR%\main.py"
    exit /b 1
)
if not exist "%FRONTEND_DIR%\package.json" (
    echo 未找到前端入口: "%FRONTEND_DIR%\package.json"
    exit /b 1
)
where python >nul 2>nul
if errorlevel 1 (
    echo 未找到 Python，请先安装 Python 并加入 PATH。
    exit /b 1
)
where node >nul 2>nul
if errorlevel 1 (
    echo 未找到 Node.js，请先安装 Node.js 并加入 PATH。
    exit /b 1
)
where npm.cmd >nul 2>nul
if errorlevel 1 (
    echo 未找到 npm，请确认 Node.js 安装完整。
    exit /b 1
)
exit /b 0

:WAIT_FOR_BACKEND
set /a BACKEND_WAIT_COUNT=0
:WAIT_FOR_BACKEND_LOOP
powershell -NoProfile -Command "try { $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2; if ($response.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>nul
if not errorlevel 1 (
    echo 后端和 MCP 已就绪。
    exit /b 0
)
set /a BACKEND_WAIT_COUNT+=1
if %BACKEND_WAIT_COUNT% GEQ 30 exit /b 1
timeout /t 1 /nobreak >nul
goto WAIT_FOR_BACKEND_LOOP

:FAILED
echo.
echo 启动失败，请查看上面的错误信息。
pause
exit /b 1
