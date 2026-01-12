import os, io, base64, re

import pdfplumber
import fitz
import cv2
import numpy as np
from PIL import Image

from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# ===================== GLOBALS =====================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CURRENT_PDF = None
all_docs = []   # هنا بنحط text chunks + figure descriptions
index = None

client = OpenAI()  # API KEY من ENV

# مودل الامبدنق
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ===================== TEXT =====================
def extract_text_clean(pdf_path):
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

    full_text = re.sub(r"-\s*\n\s*", "", full_text)
    full_text = re.sub(r"\s+", " ", full_text).strip()
    full_text = re.sub(r"\n\s*\n+", "\n\n", full_text)
    return full_text


def chunk_text(text, chunk_size=600, overlap=120):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks


# ===================== VISION =====================
def render_pages(pdf_path, dpi=200):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pages = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pages.append({"page": i + 1, "image": pix.tobytes("png")})
    return pages


def detect_visual_blocks(page_img):
    img = cv2.imdecode(np.frombuffer(page_img, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh > 0.03 * w * h:
            boxes.append((x, y, bw, bh))
    return boxes, img


def describe_image(image_bytes, context: str | None = None):
    image_b64 = base64.b64encode(image_bytes).decode()

    user_content = [
        {
            "type": "text",
            "text": (
                "You are analyzing a figure from a scientific paper.\n"
                f"Context from the paper:\n{context or '(no extra context)'}\n\n"
                "Describe ONLY the scientific content of this figure or table "
                "(axes, labels, regions, distributions, trends, etc.).\n"
                "If the image looks like a photo of a person, a face, or anything "
                "that is not a scientific figure/table/diagram from the paper, "
                "reply exactly with:\n"
                "\"This image does not appear to be a scientific figure from the paper.\""
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        },
    ]

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=400,
    )
    return res.choices[0].message.content


# ===================== RAG: INDEX BUILDING =====================
def build_index(pdf_path):
    """
    يبني الإندكس للـ PDF:
    - يستخرج النص وينظفه ويقسمه chunks
    - يكتشف البلوكات البصرية (figures/tables) ويوصفها نصيًا
    - يدمجهم كلهم في all_docs
    - يحسب لهم امبدنق ويبني FAISS index
    """
    global CURRENT_PDF, all_docs, index

    CURRENT_PDF = pdf_path
    all_docs = []

    # ---------- 1) نص ----------
    full_text = extract_text_clean(pdf_path)
    text_chunks = chunk_text(full_text)

    for c in text_chunks:
        all_docs.append({
            "type": "text",
            "content": c,
        })

    # ---------- 2) صور (figures/tables) ----------
    pages = render_pages(pdf_path)
    for p in pages:
        boxes, img = detect_visual_blocks(p["image"])
        for (x, y, w, h) in boxes:
            crop = img[y:y+h, x:x+w]
            _, buf = cv2.imencode(".png", crop)
            crop_bytes = buf.tobytes()

            # وصف الفيقير كنص (بدون ما نطلب من المستخدم يسميها)
            fig_desc = describe_image(crop_bytes, context=None)

            all_docs.append({
                "type": "figure",
                "content": fig_desc,
                "page": p["page"],  # معلومات زيادة لو احتجتيها، المستخدم ما يشوفها
            })

    # ---------- 3) امبدنق + FAISS ----------
    texts_for_emb = [d["content"] for d in all_docs]
    if not texts_for_emb:
        raise ValueError("No text or figures extracted from PDF.")

    embeddings = embed_model.encode(texts_for_emb, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # ترجعين العدد الكلي للدوكيومنتس (نص+فيقرز)
    return len(all_docs)


def get_context(question, k=5):
    """يرجع أقرب مقاطع (نص/فيقر) للسؤال من الإندكس."""
    if index is None:
        raise ValueError("PDF not processed yet")

    q_emb = embed_model.encode([question])
    _, ids = index.search(q_emb, k)
    return "\n\n".join(all_docs[i]["content"] for i in ids[0])


SYSTEM_PROMPT = """
 You are a research assistant explaining a scientific paper.

Your behavior rules:
- Your primary goal is to help the user understand the paper, clarify unclear concepts, and explain the purpose and reasoning behind components of the paper, not merely to describe what sections or elements contain.
- Answer directly and clearly.
- Use a natural academic tone.
- Before answering, make sure all the information matches what is in the paper.
- Do NOT mention phrases such as:
  "according to the context", "based on the provided text", or similar.
- Do NOT refer to the retrieval process.
- When the user asks about an algorithm, method, or technique, interpret the question at the level of the paper’s methodological approach, even if specific terminology is not used verbatim.
- Avoid dismissing a question solely because a term is not explicitly stated; instead, explain the relevant approach or mechanism described in the paper.
- When identifying the problem a paper addresses, first determine whether the paper focuses on a specific domain or on a general methodological challenge, and frame the problem accordingly.
- Before answering any high-level question (e.g., about the paper’s problem, contribution, or goal), first infer the paper’s primary focus and scope based on its overall content, then frame the answer accordingly.
- Do not assume the paper’s domain based solely on benchmarks or examples; distinguish between the core research problem and the evaluation domains used.
- Treat references to figures, tables, sections, examples, or concepts as case-insensitive and semantically equivalent (e.g., "Figure 6", "figure 6", "Fig. 6"), and preserve the intended reference across follow-up questions.
- When responding, prioritize identifying what the user is trying to understand or is confused about, and address that directly, rather than only restating the paper’s content.
- When discussing limitations or failure cases, restrict the explanation to constraints or limitations explicitly implied or discussed in the paper, and avoid generic limitations of large language models unless stated.
- When explaining why an approach is chosen over simpler alternatives, frame the explanation in terms of the core research challenge the paper aims to address, not implementation convenience or auxiliary mechanisms.
- When explaining why an approach is chosen over simpler alternatives, interpret "approach" as the paper’s core methodological or training framework, not auxiliary mechanisms such as prompting strategies or tooling details.
- If the user states an interpretation of the paper that conflicts with its core contribution, do not agree by default; instead, explicitly clarify or correct the interpretation before continuing the discussion.
- When explaining complex concepts, start with a high-level intuitive explanation suitable for graduate students, then refine it to the paper’s precise technical meaning.

Figures, tables, and boxed content:
- Treat figures, tables, and boxed examples as integral parts of the paper.
- When asked about them, explain their purpose, role, and what they demonstrate within the paper.
- Do NOT speculate beyond what is explicitly shown or described.
- If the question is phrased as "what is in Figure/Table X",
  treat it as a visual explanation task.Base the answer strictly on the figure/table description,and do not introduce additional interpretation or paper-level reasoning.
- When explaining a box or example, explicitly state why it is included in the paper and what it helps clarify, not only what it describes.
- When explaining figures, focus on what the figure demonstrates in support of the paper’s claims, not merely on visual or descriptive details.
- When explaining a figure, ensure the explanation strictly matches the specific figure number and content, and do not conflate it with other figures in the paper.

Conversational depth:
- Treat the interaction as an ongoing academic discussion rather than isolated Q&A.
- If the user asks follow-up questions (e.g., "explain more", "clarify", "go deeper"),
  continue building on the previous explanation.
- Expand by adding depth, conceptual clarification, or connections within the paper.
- Avoid repeating the same explanation verbatim; refine and deepen it instead.
- When a follow-up request is ambiguous (e.g., "explain it more"), maintain the same reference as the immediately preceding answer. If multiple interpretations are possible, ask for clarification before introducing new concepts or elements.
- For any follow-up request that asks for elaboration, clarification, or deeper explanation,
  treat the referenced element in the immediately preceding response as fixed and locked.
  Do not switch to a different figure, table, section, example, or concept unless the user
  explicitly requests a different reference. If multiple interpretations are genuinely possible,
  ask for clarification before continuing.
- Adapt the depth and style of explanation based on the user’s questions. If the user asks follow-up questions, assume they are refining their understanding rather than requesting repetition.
- If the user’s question reflects a possible misunderstanding of the paper, gently clarify or correct it using the paper’s content, without dismissing the question.
- When clarification would significantly improve the discussion, ask a concise follow-up question before continuing.

When explaining contributions:

- Always answer ONLY using the given CONTEXT of the paper.
- Be specific to THIS paper (task, data, method, and findings). 
- Never give generic answers that could fit any paper (e.g., 
  "the paper presents a novel approach" without saying what it does).
- When asked "What is the main contribution of this paper?" or similar:
  * List 2–3 bullet points.
  * Each bullet must mention:
    - what was proposed / studied (model, dataset, framework, analysis, etc.),
    - and why it matters (improvement, insight, or application).
  * Keep the whole answer under 80 words.
  
Otherwise:
- Answer the question normally using the given information.
"""



def ask_llm(question: str, context: str | None = None) -> str:
    """
    وقت السؤال:
    - لو ما انرسل context يدويًا → نجيب كونتكست من الإندكس (نص+فيقر)
    - نمرر السؤال + الكونتكست للـ LLM
    """
    if context is None:
        try:
            context = get_context(question, k=5) if index is not None else None
        except Exception:
            context = None

    if context:
        user_content = (
            f"Context from the paper:\n{context}\n\n"
            f"Question: {question}\n\n"
        )
    else:
        user_content = question

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=400,
    )

    return res.choices[0].message.content
