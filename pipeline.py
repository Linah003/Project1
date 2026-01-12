append({
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
 (خلي باقي البرومبت زي ما هو عندك)
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
