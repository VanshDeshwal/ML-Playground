#!/bin/bash

# Azure Web App deployment script for ML Playground Frontend
echo "Starting ML Playground Frontend deployment..."

# Navigate to frontend directory
cd frontend

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start the Streamlit application
echo "Starting Streamlit server..."
streamlit run app.py --server.port 8000 --server.address 0.0.0.0
