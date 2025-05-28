import os
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import json
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import numpy as np
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='rag_engine.log'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/face_recognition")
engine = create_engine(DATABASE_URL)

# Initialize HuggingFace model and tokenizer
model_name = "gpt2"  # You can use a larger model if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create text generation pipeline
text_generation = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100,
    temperature=0.7,
    device=0 if torch.cuda.is_available() else -1
)

# Initialize LangChain components
embeddings = HuggingFaceEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
llm = HuggingFacePipeline(pipeline=text_generation)

class FaceRegistrationStore:
    def __init__(self):
        self.vector_store = None
        self.last_update = None
    
    def get_registration_data(self) -> List[str]:
        """Retrieve face registration data from database"""
        with engine.connect() as connection:
            # Get registration data with face recognition confidence scores
            result = connection.execute(text("""
                SELECT 
                    fe.name,
                    fe.registered_at,
                    fv.created_at as vector_created_at
                FROM face_encodings fe
                JOIN face_vectors fv ON fe.id = fv.face_id
                ORDER BY fe.registered_at DESC
            """))
            
            documents = []
            for row in result:
                doc = (
                    f"Person named {row.name} was registered at "
                    f"{row.registered_at.strftime('%Y-%m-%d %H:%M:%S')}. "
                    f"Their face encoding was processed at "
                    f"{row.vector_created_at.strftime('%Y-%m-%d %H:%M:%S')}."
                )
                documents.append(doc)
        return documents

    def update_if_needed(self):
        """Update vector store if database has changed"""
        with engine.connect() as connection:
            result = connection.execute(text(
                "SELECT MAX(registered_at) as last_update FROM face_encodings"
            )).first()
            
            if not result.last_update:
                return
            
            if self.last_update is None or result.last_update > self.last_update:
                self.update_vector_store()
                self.last_update = result.last_update

    def update_vector_store(self):
        """Update the FAISS vector store with latest registration data"""
        # Get registration data
        documents = self.get_registration_data()
        
        # Split documents
        texts = text_splitter.split_text("\n".join(documents))
        
        # Create or update vector store
        self.vector_store = FAISS.from_texts(texts, embeddings)
        logger.info("Vector store updated with latest registration data")

    def get_qa_chain(self):
        """Create a question-answering chain"""
        if not self.vector_store:
            self.update_vector_store()
            
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True
        )

# Initialize face registration store
face_store = FaceRegistrationStore()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle RAG queries through WebSocket"""
    await websocket.accept()
    try:
        while True:
            # Receive question from client
            data = await websocket.receive_text()
            question = json.loads(data)["question"]
            
            logger.info(f"Received question: {question}")
            
            try:
                # Update vector store if needed
                face_store.update_if_needed()
                
                # Create QA chain and get response
                qa_chain = face_store.get_qa_chain()
                response = qa_chain({"query": question})
                
                # Send response back to client
                await websocket.send_json({
                    "answer": response["result"],
                    "source_documents": [str(doc) for doc in response["source_documents"]]
                })
                
                logger.info(f"Sent answer for question: {question}")
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                await websocket.send_json({
                    "error": f"Error processing your question: {str(e)}"
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 