@echo off
echo 🤖 Starting ML Playground...
echo ==================================================

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call "venv\Scripts\activate.bat"
) else (
    echo ⚠️  No virtual environment found. Using system Python...
)

REM Start using the Python script
echo 🚀 Starting servers...
python run.py

pause
