#!/bin/bash

# Start both frontend and backend servers

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Accent Detector Application...${NC}"

# Start backend server
echo -e "${BLUE}Starting FastAPI backend server...${NC}"
cd backend
python -m venv venv 2>/dev/null || echo "Virtual environment already exists"
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo -e "${GREEN}Backend server started at http://localhost:8000${NC}"

# Wait a moment for backend to initialize
sleep 2

# Start frontend server
echo -e "${BLUE}Starting React frontend server...${NC}"
cd ../frontend
npm install
npm start &
FRONTEND_PID=$!
echo -e "${GREEN}Frontend server started at http://localhost:3000${NC}"

echo -e "${GREEN}Both servers are now running!${NC}"
echo -e "${GREEN}Access the application at http://localhost:3000${NC}"
echo -e "${BLUE}Press Ctrl+C to stop both servers${NC}"

# Handle script termination
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT

# Keep script running
wait
