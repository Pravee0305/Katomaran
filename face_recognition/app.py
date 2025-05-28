from __future__ import annotations

import os
from typing import List, Dict, Optional
import cv2
import face_recognition
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, select
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import base64
import logging
import faiss
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import io
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='face_recognition.log'
)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_DIM = 512
USE_DIMENSION_REDUCTION = True

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/face_recognition")
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# FastAPI lifespan for cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_encodings_from_db()
    yield
    # Shutdown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FaceEncoding(Base):
    __tablename__ = "face_encodings"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    registered_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Initialize model with modern PyTorch practices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
features_layer = torch.nn.Sequential(*list(model.children())[:-1])

class DimensionReduction(torch.nn.Module):
    def __init__(self, input_dim: int = 512, output_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.bn = torch.nn.BatchNorm1d(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x

if USE_DIMENSION_REDUCTION:
    reduction_layer = DimensionReduction().to(device)
    model = torch.nn.Sequential(features_layer, reduction_layer)
else:
    model = features_layer

model = model.to(device)
model.eval()

# Modern image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FAISS setup with modern practices
face_index = faiss.IndexFlatL2(EMBEDDING_DIM)
if USE_DIMENSION_REDUCTION and faiss.get_num_gpus() > 0:
    face_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, face_index)

face_names: List[str] = []

def get_db() -> Session:
    """Get database session with context management"""
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    try:
        yield db
    finally:
        db.close()

def extract_face_embedding(image_array: np.ndarray) -> np.ndarray:
    """Extract face embedding using ResNet18 with optional dimension reduction"""
    try:
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Detect face location
        face_locations = face_recognition.face_locations(rgb_img)
        if not face_locations:
            raise ValueError("No face detected in the image")
        
        # Extract face ROI (Region of Interest)
        top, right, bottom, left = face_locations[0]
        face_image = rgb_img[top:bottom, left:right]
        
        # Convert to PIL Image
        pil_image = Image.fromarray(face_image)
        
        # Preprocess for ResNet
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.inference_mode():  # Modern replacement for torch.no_grad()
            embedding = model(input_batch)
        
        # Convert to numpy and flatten
        embedding_np = embedding.cpu().numpy().flatten()
        
        # Normalize the embedding
        embedding_np = embedding_np / np.linalg.norm(embedding_np)
        
        return embedding_np.astype(np.float32)
    except Exception as e:
        logger.error(f"Error extracting face embedding: {e}")
        raise

def load_encodings_from_db() -> None:
    """Load face encodings from database into FAISS index"""
    global face_index, face_names
    db = next(get_db())
    try:
        # Get all face encodings from database using modern SQLAlchemy syntax
        stmt = select(FaceEncoding.id, FaceEncoding.name).join(
            "face_vectors", FaceEncoding.id == text("face_vectors.face_id")
        )
        result = db.execute(stmt)
        
        # Collect encodings and names
        encodings = []
        face_names.clear()
        
        for row in result:
            encoding_bytes = base64.b64decode(row.encoding)
            encoding_array = np.frombuffer(encoding_bytes, dtype=np.float32)
            encodings.append(encoding_array)
            face_names.append(row.name)
        
        if encodings:
            # Convert list of encodings to numpy array
            encodings_array = np.vstack(encodings).astype(np.float32)
            
            # Reset and rebuild FAISS index
            face_index.reset()
            face_index.add(encodings_array)
            
        logger.info(f"Loaded {len(face_names)} face encodings into FAISS index")
    except Exception as e:
        logger.error(f"Error loading encodings into FAISS: {e}")
        raise
    finally:
        db.close()

@app.post("/register")
async def register_face(name: str, file: UploadFile = File(...)):
    """Register a new face"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract face embedding
        face_embedding = extract_face_embedding(img)
        
        # Store in database and FAISS
        db = next(get_db())
        try:
            # Store metadata in PostgreSQL
            db_face = FaceEncoding(name=name)
            db.add(db_face)
            db.flush()
            
            # Store vector in separate table
            encoding_bytes = base64.b64encode(face_embedding.tobytes()).decode('utf-8')
            db.execute(
                """
                INSERT INTO face_vectors (face_id, encoding) 
                VALUES (:face_id, :encoding)
                """,
                {"face_id": db_face.id, "encoding": encoding_bytes}
            )
            
            db.commit()
            
            # Update FAISS index
            face_index.add(face_embedding.reshape(1, -1))
            face_names.append(name)
            
            logger.info(f"Successfully registered face for {name}")
            return {"message": f"Successfully registered {name}"}
        finally:
            db.close()
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error registering face: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle real-time face recognition through WebSocket"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Decode base64 image
            img_data = base64.b64decode(frame_data['image'].split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            try:
                # Extract face embedding
                face_embedding = extract_face_embedding(frame)
                
                # Search in FAISS index
                distances, indices = face_index.search(face_embedding.reshape(1, -1), 1)
                
                # Get face location for bounding box
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                
                # Initialize list for face data
                face_data = []
                
                if face_locations and indices[0][0] != -1:
                    top, right, bottom, left = face_locations[0]
                    name = "Unknown"
                    confidence = 0.0
                    
                    # FAISS returns L2 distance, convert to similarity score
                    if distances[0][0] < 100:  # Threshold for face similarity
                        name = face_names[indices[0][0]]
                        confidence = float(1.0 - (distances[0][0] / 200.0))  # Normalize distance to 0-1
                    
                    face_data.append({
                        "name": name,
                        "location": {
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                            "left": left
                        },
                        "confidence": confidence
                    })
                
                # Send face data back to client
                await websocket.send_json({"faces": face_data})
                
            except ValueError:
                # No face detected in frame
                await websocket.send_json({"faces": []})
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                await websocket.send_json({"error": str(e)})
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 