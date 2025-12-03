#!/bin/bash

# Setup checker for ShuTong React Frontend
# Verifies all dependencies and configurations are correct

echo "🔍 Checking ShuTong Frontend Setup..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
ALL_OK=true

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Python found: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python not found"
    ALL_OK=false
fi

# Check Node.js
echo "Checking Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓${NC} Node.js found: $NODE_VERSION"
else
    echo -e "${RED}✗${NC} Node.js not found"
    echo -e "${YELLOW}  Install from: https://nodejs.org/${NC}"
    ALL_OK=false
fi

# Check npm
echo "Checking npm..."
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓${NC} npm found: v$NPM_VERSION"
else
    echo -e "${RED}✗${NC} npm not found"
    ALL_OK=false
fi

echo ""
echo "Checking Python dependencies..."

# Check Flask
if uv run python -c "import flask" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Flask installed"
else
    echo -e "${RED}✗${NC} Flask not installed"
    echo -e "${YELLOW}  Run: uv pip install flask flask-cors${NC}"
    ALL_OK=false
fi

# Check Flask-CORS
if uv run python -c "import flask_cors" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Flask-CORS installed"
else
    echo -e "${RED}✗${NC} Flask-CORS not installed"
    echo -e "${YELLOW}  Run: uv pip install flask-cors${NC}"
    ALL_OK=false
fi

# Check LangChain
if uv run python -c "import langchain" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} LangChain installed"
else
    echo -e "${YELLOW}⚠${NC} LangChain not found (required for problem generation)"
fi

echo ""
echo "Checking frontend setup..."

# Check if frontend directory exists
if [ -d "frontend" ]; then
    echo -e "${GREEN}✓${NC} Frontend directory exists"
    
    # Check package.json
    if [ -f "frontend/package.json" ]; then
        echo -e "${GREEN}✓${NC} package.json found"
    else
        echo -e "${RED}✗${NC} package.json not found"
        ALL_OK=false
    fi
    
    # Check node_modules
    if [ -d "frontend/node_modules" ]; then
        echo -e "${GREEN}✓${NC} node_modules installed"
    else
        echo -e "${YELLOW}⚠${NC} node_modules not installed"
        echo -e "${YELLOW}  Run: cd frontend && npm install${NC}"
        ALL_OK=false
    fi
    
    # Check .env file
    if [ -f "frontend/.env" ]; then
        echo -e "${GREEN}✓${NC} .env configuration found"
    else
        echo -e "${YELLOW}⚠${NC} .env file not found (using defaults)"
        echo -e "${YELLOW}  Copy frontend/.env.example to frontend/.env to customize${NC}"
    fi
else
    echo -e "${RED}✗${NC} Frontend directory not found"
    ALL_OK=false
fi

echo ""
echo "Checking project structure..."

# Check api_server.py
if [ -f "api_server.py" ]; then
    echo -e "${GREEN}✓${NC} api_server.py found"
else
    echo -e "${RED}✗${NC} api_server.py not found"
    ALL_OK=false
fi

# Check agent_sys
if [ -d "agent_sys" ]; then
    echo -e "${GREEN}✓${NC} agent_sys directory found"
else
    echo -e "${YELLOW}⚠${NC} agent_sys directory not found"
fi

# Check results directory
if [ -d "results" ]; then
    echo -e "${GREEN}✓${NC} results directory found"
    RESULT_COUNT=$(find results -name "result_*.json" 2>/dev/null | wc -l)
    echo -e "  Found $RESULT_COUNT result files"
else
    echo -e "${YELLOW}⚠${NC} results directory not found (charts will use mock data)"
fi

echo ""
echo "Checking ports availability..."

# Check if port 8000 is available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} Port 8000 is already in use"
    echo -e "${YELLOW}  Stop the service or use a different port${NC}"
else
    echo -e "${GREEN}✓${NC} Port 8000 is available (API server)"
fi

# Check if port 3000 is available
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} Port 3000 is already in use"
    echo -e "${YELLOW}  Stop the service or use PORT=3001 npm start${NC}"
else
    echo -e "${GREEN}✓${NC} Port 3000 is available (React dev server)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "You're ready to start the application:"
    echo ""
    echo -e "  ${GREEN}./start_frontend.sh${NC}"
    echo ""
    echo "Or manually:"
    echo ""
    echo "  Terminal 1: python api_server.py"
    echo "  Terminal 2: cd frontend && npm start"
    echo ""
else
    echo -e "${RED}✗ Some checks failed${NC}"
    echo ""
    echo "Please fix the issues above before running the application."
    echo ""
    echo "Quick fix commands:"
    echo ""
    echo "  # Install Python dependencies"
    echo "  uv pip install flask flask-cors"
    echo ""
    echo "  # Install Node dependencies"
    echo "  cd frontend && npm install"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
