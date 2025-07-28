@echo off
echo Starting ML Playground locally...
echo.

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing dependencies!
    pause
    exit /b 1
)

echo.
echo [2/3] Starting backend API...
start "ML Backend" cmd /k "cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

echo.
echo [3/3] Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo.
echo [4/4] Starting frontend...
start "ML Frontend" cmd /k "cd frontend && streamlit run app.py --server.port 8501"

echo.
echo ============================================
echo   ML Playground is starting up!
echo ============================================
echo   Backend API: http://localhost:8000
echo   Frontend:    http://localhost:8501
echo ============================================
echo.
echo Both services are starting in separate windows.
echo You can close this window once both are running.
pause
