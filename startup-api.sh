#!/bin/bash
echo "Starting ML Playground API..."
cd /home/site/wwwroot/backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
