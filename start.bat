@echo off
title Nifty SMC Options Intelligence
color 0A
echo.
echo  =========================================
echo   NIFTY SMC OPTIONS INTELLIGENCE ENGINE
echo   Smart Money Concepts + Order Flow + GEX
echo  =========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: Install requirements if needed
echo [1/3] Checking Python dependencies...
pip install -r backend\requirements.txt -q --no-warn-script-location
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [2/3] Starting FastAPI backend (port 8000)...
start "SMC Backend" cmd /k "cd backend && python -m uvicorn main:app --reload --port 8000"

echo [3/3] Waiting for backend to initialize...
timeout /t 4 /nobreak >nul

echo.
echo  [OK] Backend running at: http://localhost:8000
echo  [OK] Health check:       http://localhost:8000/health
echo  [OK] API endpoint:       http://localhost:8000/api/nifty-analysis
echo.
echo  Opening dashboard...
start "" "frontend\index.html"

echo.
echo  Dashboard is open in your browser.
echo  The backend auto-refreshes when you modify backend\main.py
echo  Press any key to exit this window (backend will keep running)
echo.
pause
