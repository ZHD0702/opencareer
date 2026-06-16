@echo off
setlocal EnableExtensions
title OpenCareer Launcher

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%web\backend"
set "FRONTEND_DIR=%ROOT%web\front"
set "FRONTEND_URL=http://localhost:5173"

if /I "%~1"=="--check" goto CHECK

echo ========================================
echo OpenCareer Launcher
echo ========================================
echo.

call :VALIDATE
if errorlevel 1 goto FAILED

echo [0/4] Stopping stale OpenCareer services...
powershell -NoProfile -Command "$connections = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue); foreach ($connection in $connections) { if ($connection.LocalPort -in @(8001, 8002, 5173)) { Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue } }" >nul 2>nul
timeout /t 1 /nobreak >nul

echo [1/4] Checking backend and MCP dependencies...
python -c "import fastapi, uvicorn, mcp" >nul 2>nul
if errorlevel 1 (
    echo Installing backend and MCP dependencies...
    python -m pip install -r "%BACKEND_DIR%\requirements.txt"
    if errorlevel 1 goto FAILED
)

echo.
echo [2/4] Checking frontend dependencies...
if not exist "%FRONTEND_DIR%\node_modules" (
    echo Installing frontend dependencies...
    pushd "%FRONTEND_DIR%"
    call npm.cmd install
    if errorlevel 1 (
        popd
        goto FAILED
    )
    popd
)

echo.
echo [3/4] Starting backend and MCP...
echo Backend: http://localhost:8002
echo MCP:     http://localhost:8001/mcp
start "OpenCareer Backend and MCP" /D "%BACKEND_DIR%" cmd /k "set CAREER_USE_MCP=true&&set CAREER_MCP_URL=http://127.0.0.1:8001/mcp&&set HOST=127.0.0.1&&set PORT=8002&&python -m uvicorn main:app --host 127.0.0.1 --port 8002"

echo Waiting for backend and MCP startup...
call :WAIT_FOR_BACKEND
if errorlevel 1 echo Backend is not ready yet. The frontend will still start.

echo.
echo [4/4] Starting frontend...
echo Frontend: %FRONTEND_URL%
start "OpenCareer Frontend" /D "%FRONTEND_DIR%" cmd /k "npm.cmd run dev -- --host 127.0.0.1 --port 5173"

timeout /t 3 /nobreak >nul
start "" "%FRONTEND_URL%"

echo.
echo ========================================
echo OpenCareer started
echo ========================================
echo Open %FRONTEND_URL% in your browser.
exit /b 0

:CHECK
call :VALIDATE
if errorlevel 1 exit /b 1
echo OpenCareer launcher check passed.
echo Root:     %ROOT%
echo Backend:  %BACKEND_DIR%
echo Frontend: %FRONTEND_DIR%
echo URL:      %FRONTEND_URL%
exit /b 0

:VALIDATE
if not exist "%BACKEND_DIR%\main.py" (
    echo Backend entry not found: "%BACKEND_DIR%\main.py"
    exit /b 1
)
if not exist "%FRONTEND_DIR%\package.json" (
    echo Frontend entry not found: "%FRONTEND_DIR%\package.json"
    exit /b 1
)
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found in PATH.
    exit /b 1
)
where node >nul 2>nul
if errorlevel 1 (
    echo Node.js was not found in PATH.
    exit /b 1
)
where npm.cmd >nul 2>nul
if errorlevel 1 (
    echo npm.cmd was not found in PATH.
    exit /b 1
)
exit /b 0

:WAIT_FOR_BACKEND
set /a BACKEND_WAIT_COUNT=0
:WAIT_FOR_BACKEND_LOOP
powershell -NoProfile -Command "try { $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8002/api/mcp/status' -TimeoutSec 4; $data = $response.Content | ConvertFrom-Json; if ($response.StatusCode -eq 200 -and $data.resume_skill_available) { exit 0 } } catch {}; exit 1" >nul 2>nul
if not errorlevel 1 (
    echo Backend, MCP, and resume_skill are ready.
    exit /b 0
)
set /a BACKEND_WAIT_COUNT+=1
if %BACKEND_WAIT_COUNT% GEQ 30 exit /b 1
timeout /t 1 /nobreak >nul
goto WAIT_FOR_BACKEND_LOOP

:FAILED
echo.
echo Startup failed. Review the errors above.
exit /b 1
