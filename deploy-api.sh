#!/bin/bash

# Azure Web App deployment script for ML Playground API
echo "Starting ML Playground API deployment..."

# Navigate to backend directory
cd backend

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start the FastAPI application
echo "Starting FastAPI server..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000
