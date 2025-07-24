#!/usr/bin/env python3
"""
Simple startup script for ML Playground
Starts both FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import os
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import fastapi
        import streamlit
        import uvicorn
        import numpy
        import pandas
        import plotly
        import sklearn
        import requests
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    cmd = [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def start_frontend():
    """Start the Streamlit frontend"""
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def wait_for_server(url, timeout=30):
    """Wait for server to be ready"""
    import requests
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def main():
    """Main function to start the ML Playground"""
    print("🤖 Starting ML Playground...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start backend
    print("🚀 Starting FastAPI backend...")
    backend_process = start_backend()
    
    # Wait for backend to be ready
    print("⏳ Waiting for backend to start...")
    if wait_for_server("http://localhost:8000"):
        print("✅ Backend is ready at http://localhost:8000")
    else:
        print("❌ Backend failed to start")
        backend_process.terminate()
        sys.exit(1)
    
    # Start frontend
    print("🎨 Starting Streamlit frontend...")
    frontend_process = start_frontend()
    
    # Wait for frontend to be ready
    print("⏳ Waiting for frontend to start...")
    if wait_for_server("http://localhost:8501"):
        print("✅ Frontend is ready at http://localhost:8501")
    else:
        print("❌ Frontend failed to start")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 ML Playground is now running!")
    print("📊 Frontend: http://localhost:8501")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("=" * 50)
    print("Press Ctrl+C to stop all servers")
    
    # Open browser
    try:
        webbrowser.open("http://localhost:8501")
    except:
        pass
    
    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ML Playground...")
        backend_process.terminate()
        frontend_process.terminate()
        print("✅ All servers stopped")

if __name__ == "__main__":
    main()
