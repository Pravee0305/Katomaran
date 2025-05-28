@echo off
echo Starting Face Recognition Platform Services...

REM Start Face Recognition Service
start cmd /k "cd face_recognition && python app.py"

REM Start RAG Engine Service
start cmd /k "cd rag_engine && python app.py"

REM Start Node.js Backend
start cmd /k "cd backend && npm start"

REM Start React Frontend
start cmd /k "cd frontend && npm start"

echo All services started!
echo Face Recognition Service: http://localhost:8000
echo RAG Engine Service: http://localhost:8001
echo Backend Service: http://localhost:3000
echo Frontend: http://localhost:3001 