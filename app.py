# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, shutil

from pipeline import build_index, ask_llm

app = FastAPI(title="Research Paper Chatbot Backend")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    num_docs = build_index(file_path)

    return {
        "status": "uploaded_and_indexed",
        "file_path": file_path,
        "docs_in_index": num_docs,
    }


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(body: AskRequest):
    answer = ask_llm(body.question)
    return {"answer": answer}
