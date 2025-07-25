#!/bin/bash

# ML Playground Frontend Startup Script for Azure Web Apps
echo "=== ML Playground Frontend Startup ==="
echo "Current directory: $(pwd)"
echo "Contents: $(ls -la)"

# Check if we're in the right location
if [ -d "frontend" ]; then
    echo "✅ Found frontend directory"
    cd frontend
    echo "📁 Switched to frontend directory: $(pwd)"
    echo "📄 Contents: $(ls -la)"
    
    # Install dependencies
    if [ -f "requirements.txt" ]; then
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
        echo "✅ Dependencies installed"
    else
        echo "❌ requirements.txt not found"
    fi
    
    # Check if app.py exists
    if [ -f "app.py" ]; then
        echo "✅ Found app.py"
        
        # Create Streamlit config directory
        mkdir -p ~/.streamlit
        
        # Create Streamlit config file
        cat > ~/.streamlit/config.toml << EOF
[server]
port = 8000
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
headless = true

[browser]
gatherUsageStats = false
EOF
        
        # Set environment variables
        export API_BASE_URL="https://ml-playground-api.azurewebsites.net"
        
        echo "🚀 Starting Streamlit server..."
        streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true
    else
        echo "❌ app.py not found in $(pwd)"
        echo "Contents: $(ls -la)"
    fi
else
    echo "❌ Frontend directory not found in $(pwd)"
    echo "Available directories: $(ls -la)"
fi
