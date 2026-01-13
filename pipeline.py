# pipeline.py
import os, io, base64, re

import pdfplumber
import fitz
import cv2
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# ===================== GLOBALS =====================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CURRENT_PDF: str | None = None
all_docs: list[dict] = []   # نخزن فيها chunks النص + أوصاف الفيقير
index: faiss.IndexFlatL2 | None = None

client = OpenAI()  # OPENAI_API_KEY من ENV

# مودل الامبدنق
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ===================== TEXT =====================
def extract_text_clean(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(
                use_text_flow=True,
                keep_blank_chars=False,
                x_tolerance=2,
                y_tolerance=2
            )
            if words:
                full_text += " ".join(w["text"] for w in words) + "\n"

    # تنظيف بسيط
    full_text = re.sub(r"-\s*\n\s*", "", full_text)
    full_text = re.sub(r"\s+", " ", full_text).strip()
    full_text = re.sub(r"\n\s*\n+", "\n\n", full_text)
    return full_text


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> list[str]:
    chunks, start = [], 0
    n = len(text)
    while start < n:
        chunks.append(text[start:start+chunk_size])
        start += max(1, chunk_size - overlap)
    return chunks


# ===================== VISION =====================
def render_pages(pdf_path: str, dpi: int = 200) -> list[dict]:
    """نحوّل كل صفحة لصورة PNG."""
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pages = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pages.append({"page": i + 1, "image": pix.tobytes("png")})
    return pages


def detect_visual_blocks(page_img: bytes):
    """نحدد البلوكات الكبيرة (شكلها فيقر/تيبل)."""
    img = cv2.imdecode(np.frombuffer(page_img, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh > 0.03 * w * h:  # نتجاهل الأشياء الصغيرة
            boxes.append((x, y, bw, bh))
    return boxes, img


def describe_image(image_bytes: bytes, context: str | None = None) -> str:
    """نسأل نموذج فيجن يشرح الفيقير كنص."""
    image_b64 = base64.b64encode(image_bytes).decode()

    user_content = [
        {
            "type": "text",
            "text": (
                "You are analyzing a figure from a scientific paper.\n"
                f"Context from the paper:\n{context or '(no extra context)'}\n\n"
                "Describe ONLY the scientific content of this figure or table "
                "(axes, labels, distributions, trends, comparisons, etc.).\n"
                "If the image is not a scientific figure/table/diagram, reply exactly:\n"
                "\"This image does not appear to be a scientific figure from the paper.\""
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        },
    ]

    res = client.chat.completions.create(
        model="gpt-4o-mini",   # لازم مودل يشوف صور
        messages=[
            {
                "role": "system",
                "content": "You describe scientific figures briefly and clearly.",
            },
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=300,
    )
    return res.choices[0].message.content


# ===================== RAG: INDEX BUILDING =====================
def build_index(pdf_path: str) -> int:
    """
    يبني الإندكس للـ PDF:
    - يستخرج النص وينظفه ويقسمه chunks
    - يكتشف الفيقير/تيبل ويحولها لوصف نصي
    - يدمج كل شيء في all_docs
    - يحسب امبدنق ويبني FAISS index
    """
    global CURRENT_PDF, all_docs, index

    CURRENT_PDF = pdf_path
    all_docs = []

    # ---------- 1) نص ----------
    full_text = extract_text_clean(pdf_path)
    text_chunks = chunk_text(full_text)

    for c in text_chunks:
        all_docs.append(
            {
                "type": "text",
                "content": c,
            }
        )

    # ---------- 2) صور (figures/tables) ----------
    pages = render_pages(pdf_path)
    for p in pages:
        boxes, img = detect_visual_blocks(p["image"])
        for (x, y, w, h) in boxes:
            crop = img[y : y + h, x : x + w]
            _, buf = cv2.imencode(".png", crop)
            crop_bytes = buf.tobytes()

            fig_desc = describe_image(crop_bytes, context=None)

            all_docs.append(
                {
                    "type": "figure",
                    "content": fig_desc,
                    "page": p["page"],
                }
            )

    # ---------- 3) امبدنق + FAISS ----------
    texts_for_emb = [d["content"] for d in all_docs]
    if not texts_for_emb:
        raise ValueError("No text or figures extracted from PDF.")

    embeddings = embed_model.encode(texts_for_emb, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print(f"[BUILD_INDEX] docs in index: {len(all_docs)}")
    return len(all_docs)


def get_context(question: str, k: int = 5) -> str:
    """نرجّع أقرب k مقاطع (نص/فيقر) للسؤال من الإندكس."""
    if index is None:
        raise ValueError("PDF not processed yet (index is None)")

    q_emb = embed_model.encode([question])
    _, ids = index.search(q_emb, k)

    parts = [all_docs[i]["content"] for i in ids[0]]
    context = "\n\n---\n\n".join(parts)
    return context


# ===================== LLM Q&A =====================
SYSTEM_PROMPT = """
You are a research assistant explaining a scientific paper.
Your job is to help the user understand the paper: clarify concepts,
explain figures, methods, contributions, and results.
Answer directly and clearly in 3–6 sentences.
Do not mention retrieval or chunks. Just answer as if you read the paper.
"""


def ask_llm(question: str) -> str:
    """يأخذ سؤال المستخدم، يجيب كونتكست من الإندكس، ويرسلهم للنموذج."""
    try:
        context = get_context(question, k=5)
    except Exception as e:
        print("[ASK_LLM] get_context error:", e)
        context = ""

    print("[ASK_LLM] CONTEXT LEN:", len(context))

    user_message = (
        "Here is relevant context from the paper:\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on this paper. "
        "If the context is empty or does not contain the answer, say that clearly."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",   # أو gpt-4o-mini عادي
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=400,
        )

        content = response.choices[0].message.content or ""
        print("[ASK_LLM] RAW CONTENT PREVIEW:", repr(content[:200]))

        if not content.strip():
            return (
                "The model returned an empty answer. "
                "Please try asking again or check that the uploaded paper contains relevant text."
            )

        return content.strip()

    except Exception as e:
        print("[ASK_LLM] OPENAI ERROR:", e)
        return f"Backend error while contacting the model: {e}"
