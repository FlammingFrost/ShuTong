#!/bin/bash

# Startup script for ShuTong React Frontend
# This script starts both the Flask API backend and React frontend

echo "🧮 Starting ShuTong Math Agent System..."
echo ""

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
if ! uv run python -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask not found. Installing Flask dependencies..."
    uv pip install flask flask-cors
fi

# Check if Node modules are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "⚠️  Node modules not found. Installing..."
    cd frontend
    npm install
    cd ..
fi

# Start the Flask API in the background
echo ""
echo "🚀 Starting Flask API server on port 8000..."
uv run python api_server.py &
API_PID=$!

# Wait for API to start
sleep 3

# Check if API started successfully
if kill -0 $API_PID 2>/dev/null; then
    echo "✅ API server started successfully (PID: $API_PID)"
else
    echo "❌ Failed to start API server"
    exit 1
fi

# Start the React development server
echo ""
echo "🚀 Starting React development server on port 3000..."
cd frontend
npm start

# When React server stops, also stop the API
echo ""
echo "🛑 Shutting down API server..."
kill $API_PID 2>/dev/null

echo "👋 Goodbye!"
