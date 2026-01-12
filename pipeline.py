[c["content"] for c in all_docs],
        convert_to_numpy=True
    )

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # 3) الصفحات + visual blocks لكل صفحة (عشان الفيقيرز / الجداول)
    PAGES = []
    raw_pages = render_pages(pdf_path)
    for p in raw_pages:
        boxes, _ = detect_visual_blocks(p["image"])
        PAGES.append({
            "page": p["page"],
            "image": p["image"],
            "boxes": boxes
        })

    return len(all_docs)


def get_context(question, k=5):
    """يرجع أقرب مقاطع للسؤال من الإندكس."""
    if index is None:
        raise ValueError("PDF not processed yet")

    q_emb = embed_model.encode([question])
    _, ids = index.search(q_emb, k)
    return "\n\n".join(all_docs[i]["content"] for i in ids[0])


SYSTEM_PROMPT = """
 You are a research assistant explaining a scientific paper.
 ... (نفس البرومبت اللي عندك بالضبط) ...
"""


def ask_llm(question: str) -> str:
    """
    يسوي:
    1) يجمع كونتكست من الورقة (RAG) عن طريق get_context
    2) يرسل السؤال + الكونتكست للـ LLM
    """
    try:
        context = get_context(question, k=5) if index is not None else None
    except Exception:
        context = None

    if context:
        user_content = (
            f"Context from the paper:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer clearly in 3–6 sentences."
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


def describe_block(page_number: int, block_index: int, question: str | None = None) -> str:
    """
    يختار بلوك بصري (مثلاً فيقر) من صفحة معيّنة:
    - يقصّه من صورة الصفحة
    - يجيب كونتكست نصّي من الورقة لو فيه سؤال
    - يمرره لـ describe_image
    """
    if not PAGES:
        raise ValueError("No pages loaded. Upload and index a PDF first.")

    page_entry = next((p for p in PAGES if p["page"] == page_number), None)
    if page_entry is None:
        raise ValueError(f"Page {page_number} not found.")

    boxes = page_entry["boxes"]
    if block_index < 0 or block_index >= len(boxes):
        raise ValueError(f"Block index {block_index} out of range for page {page_number}.")

    x, y, w, h = boxes[block_index]

    img = cv2.imdecode(np.frombuffer(page_entry["image"], np.uint8), cv2.IMREAD_COLOR)
    crop = img[y:y+h, x:x+w]
    _, buf = cv2.imencode(".png", crop)
    crop_bytes = buf.tobytes()

    ctx = None
    if question:
        try:
            ctx = get_context(question, k=5)
        except Exception:
            ctx = None

    return describe_image(crop_bytes, context=ctx)
