from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

from pipeline import (
    build_index,
    ask_llm,
)

app = FastAPI(title="Research Paper Chatbot Backend")

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # حطي رابط موقعك بدل * لو حابة
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== UPLOADS FOLDER =====
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ===== ENDPOINT: رفع الـ PDF وبناء البايب لاين (نص + صور) =====
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # نحفظ الملف
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # نبني الإندكس (نص + وصف الفيقيرز) مره وحده
        num_docs = build_index(file_path)

        return {
            "status": "uploaded_and_indexed",
            "file_path": file_path,
            "docs_in_index": num_docs,   # نص + فيقرز
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== ENDPOINT: سؤال الشات بوت =====
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """
    يستقبل سؤال المستخدم عن الورقة:
    - RAG في البايب لاين يجيب كونتكست من النص + الفيقيرز
    - LLM يجاوب بناءً على الكونتكست
    """
    try:
        answer = ask_llm(question)
        return {
            "question": question,
            "answer": answer,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
