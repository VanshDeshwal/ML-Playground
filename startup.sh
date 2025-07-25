#!/bin/bash

# ML Playground API Startup Script for Azure Web Apps
echo "=== ML Playground API Startup ==="
echo "Current directory: $(pwd)"
echo "Contents: $(ls -la)"

# Check if we're in the right location
if [ -d "backend" ]; then
    echo "✅ Found backend directory"
    cd backend
    echo "📁 Switched to backend directory: $(pwd)"
    echo "📄 Contents: $(ls -la)"
    
    # Install dependencies
    if [ -f "requirements.txt" ]; then
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
        echo "✅ Dependencies installed"
    else
        echo "❌ requirements.txt not found"
    fi
    
    # Check if main.py exists
    if [ -f "main.py" ]; then
        echo "✅ Found main.py"
        echo "🚀 Starting FastAPI server..."
        python -m uvicorn main:app --host 0.0.0.0 --port 8000
    else
        echo "❌ main.py not found in $(pwd)"
        echo "Contents: $(ls -la)"
    fi
else
    echo "❌ Backend directory not found in $(pwd)"
    echo "Available directories: $(ls -la)"
fi
