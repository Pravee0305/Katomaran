import face_recognition
import cv2
import numpy as np
import json
import asyncio
import websockets
import sqlite3
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceRecognitionService:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.db_path = 'database/faces.db'
        self.setup_database()
        self.load_known_faces()

    def setup_database(self):
        """Initialize SQLite database"""
        os.makedirs('database', exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS faces
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             name TEXT NOT NULL,
             encoding BLOB NOT NULL,
             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
        ''')
        conn.commit()
        conn.close()

    def load_known_faces(self):
        """Load known faces from database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT name, encoding FROM faces')
        rows = c.fetchall()
        for row in rows:
            name, encoding = row
            self.known_face_names.append(name)
            self.known_face_encodings.append(np.frombuffer(encoding))
        conn.close()
        logger.info(f"Loaded {len(self.known_face_names)} faces from database")

    async def register_face(self, frame_data, name):
        """Register a new face"""
        try:
            # Convert frame data to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(frame)
            if not face_locations:
                return {"success": False, "message": "No face detected"}
            
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            if not face_encodings:
                return {"success": False, "message": "Could not encode face"}

            # Store in database
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('INSERT INTO faces (name, encoding) VALUES (?, ?)',
                     (name, face_encodings[0].tobytes()))
            conn.commit()
            conn.close()

            # Update in-memory lists
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            
            logger.info(f"Successfully registered face for {name}")
            return {"success": True, "message": f"Successfully registered {name}"}

        except Exception as e:
            logger.error(f"Error registering face: {str(e)}")
            return {"success": False, "message": str(e)}

    async def recognize_faces(self, frame_data):
        """Recognize faces in frame"""
        try:
            # Convert frame data to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Find faces in frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Compare face with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"
                    
                face_names.append(name)

            # Return results
            results = []
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                results.append({
                    "name": name,
                    "location": {
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "left": left
                    }
                })
            
            return {"success": True, "faces": results}

        except Exception as e:
            logger.error(f"Error recognizing faces: {str(e)}")
            return {"success": False, "message": str(e)}

async def websocket_handler(websocket, path):
    """Handle WebSocket connections"""
    face_service = FaceRecognitionService()
    logger.info("New WebSocket connection established")

    try:
        async for message in websocket:
            data = json.loads(message)
            action = data.get('action')
            
            if action == 'register':
                result = await face_service.register_face(
                    data['frame'],
                    data['name']
                )
            elif action == 'recognize':
                result = await face_service.recognize_faces(data['frame'])
            else:
                result = {"success": False, "message": "Invalid action"}
            
            await websocket.send(json.dumps(result))

    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Error in websocket handler: {str(e)}")

if __name__ == "__main__":
    start_server = websockets.serve(
        websocket_handler,
        "localhost",
        8765
    )

    logger.info("Starting Face Recognition Service on ws://localhost:8765")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever() 