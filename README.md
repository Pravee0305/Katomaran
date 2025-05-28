# Face Recognition Platform with Real-Time AI Q&A

A browser-based platform for face registration and recognition with RAG-powered chat interface.

## Architecture
```
                                    +-------------------+
                                    |     Frontend     |
                                    |     (React)      |
                                    +--------+---------+
                                             |
                                             v
                                    +-------------------+
                                    |   Node.js API    |
                                    |  WebSocket Server|
                                    +--------+---------+
                                             |
                        +--------------------+--------------------+
                        v                    v                   v
                +---------------+    +--------------+    +---------------+
                |Face Detection |    |  RAG Engine  |    |   Database   |
                |   (Python)    |    |  (Python)    |    | (PostgreSQL) |
                +---------------+    +--------------+    +---------------+

## Features
- Real-time face registration and recognition
- Multi-face detection support
- RAG-powered chat interface for querying face registration data
- WebSocket-based real-time communication

## Tech Stack
- Frontend: React.js
- Backend: Node.js
- Face Recognition: Python (OpenCV)
- RAG: LangChain + FAISS + Hugging Face GPT
- Database: PostgreSQL
- WebSocket: Socket.io

## Project Structure
```
face-recognition-platform/
├── frontend/               # React frontend
├── backend/               # Node.js backend
│   ├── api/              # REST API endpoints
│   └── websocket/        # WebSocket server
├── face_recognition/     # Python face recognition service
├── rag_engine/          # Python RAG implementation
└── database/            # Database schemas and migrations
```

## Setup Instructions

1. Clone the repository
```bash
git clone <repository-url>
cd face-recognition-platform
```

2. Install dependencies

Frontend:
```bash
cd frontend
npm install
```

Backend:
```bash
cd backend
npm install
```

Python services:
```bash
cd face_recognition
pip install -r requirements.txt

cd ../rag_engine
pip install -r requirements.txt
```

3. Set up environment variables
Create `.env` files in each service directory with the required configuration.

4. Start the services
```bash
# Start frontend
cd frontend
npm start

# Start backend
cd backend
npm start

# Start Python services
cd face_recognition
python app.py

cd rag_engine
python app.py
```

## API Documentation
[API documentation will be added here]

## Contributing
[Contributing guidelines will be added here]

This project is a part of a hackathon run by https://katomaran.com 