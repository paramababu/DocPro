from .embedder import embed_texts, cosine_sim

def search(query, chunk_vectors, chunks, top_k):
    qvec = embed_texts([query])[0]
    scores = []
    for i, cvec in enumerate(chunk_vectors):
        s = cosine_sim(qvec, cvec)
        scores.append((chunks[i], s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def build_prompt(contexts, question):
    blocks = [f"[Context {i+1} | score={s:.3f}]\n{text}" for i, (text, s) in enumerate(contexts)]
    return (
        "Answer the question using ONLY the context below. "
        "If the answer isn’t there, say you don’t know.\n\n"
        f"Context:\n\n{chr(10).join(blocks)}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
