# Research Paper Chatbot – Backend (RAG Core)

Minimal README for the PDF → RAG → QA part of the project.

## Files

- app.py
  - POST /upload → calls build_index
  - POST /ask → calls ask_llm
- pipeline.py
  - PDF processing, figure detection, FAISS index, and LLM calls
- main.py
  - Entry point: from app import app (used by `uvicorn`)

---

## High-level Flow

1. Client uploads a PDF to POST /upload.
2. upload_pdf saves it to uploads/ and calls build_index(pdf_path).
3. build_index:
   - Extracts and chunks text.
   - Detects figures/tables, describes them.
   - Builds embeddings + FAISS index into globals: CURRENT_PDF, all_docs, index.
4. Client sends a question to POST /ask.
5. ask_question calls ask_llm(question):
   - Retrieves relevant chunks with get_context.
   - Sends context + question to gpt-4o-mini.
   - Returns the answer.

---

## Endpoints (app.py)

- POST /upload
  - Input: file (PDF, multipart/form-data)
  - Uses: build_index(file_path)
  - Output JSON: { status, file_path, docs_in_index }

- POST /ask
  - Input JSON: { "question": "<user question>" }
  - Uses: ask_llm(question)
  - Output JSON: { "answer": "<model answer>" }

---

## Core Methods (pipeline.py)

- extract_text_clean(pdf_path: str) -> str  
  Input: PDF file path  
  Does: Reads all pages with pdfplumber, cleans whitespace/hyphens  
  Output: Single string with the paper’s text

- chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> list[str]  
  Input: Full text string  
  Does: Splits text into overlapping chunks for retrieval  
  Output: list[str] of text chunks

- render_pages(pdf_path: str, dpi: int = 200) -> list[dict]  
  Input: PDF file path  
  Does: Renders each page as an image using PyMuPDF  
  Output: List of { "page": int, "image": bytes }

- detect_visual_blocks(page_img: bytes)  
  Input: Page image bytes (PNG)  
  Does: Uses OpenCV to detect large visual regions (figures/tables)  
  Output: (boxes, img) where boxes is a list of (x, y, w, h) and img is the OpenCV image

- describe_image(image_bytes: bytes, context: str | None = None) -> str  
  Input: Cropped figure image bytes, optional text context  
  Does: Sends image + instructions to gpt-4o-mini (vision) to get a scientific description  
  Output: Short text description of the figure/table (or a fixed message if not a scientific figure)

- build_index(pdf_path: str) -> int  
  Input: PDF file path  
  Does:  
  - Resets globals: CURRENT_PDF, all_docs, index  
  - Extracts text and chunks it (`extract_text_clean`, chunk_text`) and stores chunks in `all_docs  
  - Renders pages, finds visual blocks, describes figures (`render_pages`, detect_visual_blocks, describe_image`) and adds them to `all_docs  
  - Builds embeddings with SentenceTransformer and a FAISS index from all content in all_docs  
  Output: Number of documents stored in the index (`len(all_docs)`)

- get_context(question: str, k: int = 5) -> str  
  Input: User question, number of neighbors k  
  Does: Encodes the question, searches FAISS index, fetches top-`k` entries from all_docs, and joins them  
  Output: Context string used for the LLM prompt

- ask_llm(question: str, context: str | None = None) -> str  
  Input: User question, optional external context  
  Does:  
  - If no context is provided and index exists, calls get_context(question)  
  - Builds a user message with context + question  
  - Calls gpt-4o-mini with a research-assistant SYSTEM_PROMPT  
  - Handles empty responses gracefully  
  Output: Final answer string returned to the API

---

## Globals

- CURRENT_PDF: str | None  
  Path of the last indexed PDF (for reference).

- all_docs: list[dict]  
  All RAG documents:
  - Text chunks: { "type": "text", "content": str }
  - Figure descriptions: { "type": "figure", "content": str, "page": int }

- index: faiss.IndexFlatL2 | None  
  FAISS index of embeddings over all_docs.
