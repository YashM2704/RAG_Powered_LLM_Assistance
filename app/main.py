from fastapi import FastAPI, UploadFile, File
from app.ingestion.ingest import ingest_documents
from pydantic import BaseModel
from app.retrieval.rag import ask_question
import shutil
import os
app = FastAPI(
    title="RAG LLM Assistant",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "RAG Assistant Running"
    }


@app.post("/ask")
def ask(request: QueryRequest):
    result = ask_question(request.question)

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }
    
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    upload_dir = "data/documents"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(
        upload_dir,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = ingest_documents()

    return {
        "message": "PDF uploaded and indexed",
        "pages": result["documents"],
        "chunks": result["chunks"]
    }