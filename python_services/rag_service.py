import asyncio
import websockets
import json
import logging
import sqlite3
from datetime import datetime
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain import FAISS, HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.db_path = 'database/faces.db'
        self.setup_models()
        self.setup_vector_store()

    def setup_models(self):
        """Initialize the language models and embeddings"""
        # Load GPT model from Hugging Face
        model_name = "gpt2"  # You can use larger models like "gpt2-medium" or "gpt2-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            max_length=100,
            temperature=0.7
        )
        
        # Create LangChain pipeline
        self.llm = HuggingFacePipeline(pipeline=model)
        
        # Setup embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    def setup_vector_store(self):
        """Initialize FAISS vector store with face registration data"""
        # Get face registration data from SQLite
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT name, timestamp FROM faces')
        rows = c.fetchall()
        conn.close()

        # Create documents for vector store
        documents = []
        for name, timestamp in rows:
            content = f"Person named {name} was registered at {timestamp}."
            documents.append(Document(page_content=content))

        # Create vector store
        self.vector_store = LangchainFAISS.from_documents(
            documents,
            self.embeddings
        )

    async def answer_query(self, query):
        """Answer questions about face registration activities"""
        try:
            # Search relevant documents
            docs = self.vector_store.similarity_search(query, k=3)
            
            # Prepare context
            context = "\n".join([doc.page_content for doc in docs])
            
            # Prepare prompt
            prompt = f"""Based on the following information about face registrations, please answer the question.

Context:
{context}

Question: {query}

Answer:"""

            # Generate response
            response = self.llm(prompt)[0]['generated_text']
            
            # Extract the answer part (after "Answer:")
            answer = response.split("Answer:")[-1].strip()
            
            return {
                "success": True,
                "answer": answer
            }

        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }

    async def update_vector_store(self):
        """Update vector store with new registration data"""
        try:
            self.setup_vector_store()
            return {"success": True, "message": "Vector store updated successfully"}
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")
            return {"success": False, "message": str(e)}

async def websocket_handler(websocket, path):
    """Handle WebSocket connections"""
    rag_service = RAGService()
    logger.info("New WebSocket connection established")

    try:
        async for message in websocket:
            data = json.loads(message)
            action = data.get('action')
            
            if action == 'query':
                result = await rag_service.answer_query(data['query'])
            elif action == 'update_store':
                result = await rag_service.update_vector_store()
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
        8766  # Different port from face recognition service
    )

    logger.info("Starting RAG Service on ws://localhost:8766")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever() 