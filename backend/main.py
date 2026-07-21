from fastapi import FastAPI, UploadFile, File
import os
import shutil
from src.pipeline.rag_pipeline import RAGPipeline
from fastapi import HTTPException
from pydantic import BaseModel
from backend.services.chat_service import ChatService

app = FastAPI(
    title="PDF RAG Chatbot API",
    description="Backend API for PDF upload, retrieval, and RAG-based question answering.",
    version="1.0.0"
)

chat_service = ChatService()

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {
        "message": "PDF RAG Chatbot API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    
    os.makedirs("data/uploads", exist_ok=True)

    file_path = os.path.join(
        "data/uploads",
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "saved_to": file_path,
        "message": "Upload successful"
    }

@app.post("/process")
def process_pdf(filename: str):
    file_path = os.path.join(
        "data/uploads",
        filename
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="File not found."
        )

    stats = chat_service.process_pdf(
        file_path
    )

    return {
        "message": "PDF processed successfully",
        "stats": stats
    }

@app.post("/chat")
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    response = chat_service.chat(
        request.question
    )

    return response