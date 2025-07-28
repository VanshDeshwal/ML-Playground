# ML Playground - Local Development Setup

Write-Host "Starting ML Playground locally..." -ForegroundColor Green
Write-Host ""

Write-Host "[1/3] Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error installing dependencies!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[2/3] Starting backend API..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; python -m uvicorn main:app --host 0.0.0.0 --port 8000"

Write-Host ""
Write-Host "[3/3] Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "[4/4] Starting frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; streamlit run app.py --server.port 8501"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   ML Playground is starting up!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "   Frontend:    http://localhost:8501" -ForegroundColor White
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Both services are starting in separate windows." -ForegroundColor Green
Write-Host "You can close this window once both are running." -ForegroundColor Green
Read-Host "Press Enter to exit"
