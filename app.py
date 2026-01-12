from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

from pipeline import build_index, ask_llm
from typing import Optional
from pydantic import BaseModel

app = FastAPI(title="Research Paper Chatbot Backend")

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # لاحقًا تقدرين تشيلين * وتحطين دومين واجهتك
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== UPLOADS FOLDER =====
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ===== ENDPOINT: رفع الـ PDF وبناء الإندكس =====
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # نحفظ الملف
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # نبني الإندكس (نص + فيقرز)
    num_docs = build_index(file_path)

    return {
        "status": "uploaded_and_indexed",
        "file_path": file_path,
        "docs_in_index": num_docs,   # عدد الدوكيومنتس في الإندكس
    }


# ===== ENDPOINT: سؤال الشات بوت =====

class AskRequest(BaseModel):
    question: str
    previous_question: Optional[str] = None
    previous_answer: Optional[str] = None
    
@app.post("/ask")
async def ask_question(body: AskRequest):
    """
    receives JSON: 
    { 
      "question": "...", 
      "previous_question": "...", 
      "previous_answer": "..." 
    }
    """
    answer = ask_llm(
        question=body.question,
        previous_question=body.previous_question,
        previous_answer=body.previous_answer
    )
    return {
        "question": body.question,
        "answer": answer,
    }

