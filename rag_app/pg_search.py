import os
import sys
import time
import logging
import ollama

# Ensure project root (parent of this folder) is on sys.path
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from rag_app.db import get_conn, blob_to_vec
from rag_app import embedder

_log = logging.getLogger("docpro.search")
if not _log.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    _log.setLevel(getattr(logging, level, logging.INFO))

def search_db(query, top_k=5):
    _log.info("search_db | top_k=%s qlen=%s", top_k, len(query))
    conn = get_conn()
    cur = conn.cursor()

    t0 = time.perf_counter()
    q_vec = embedder.embed_texts([query])[0]
    _log.info("embedded query | dim=%s elapsed=%.3fs", len(q_vec), time.perf_counter() - t0)

    cur.execute(
        """
        SELECT files.path, chunks.content, chunks.embedding
        FROM chunks
        JOIN files ON chunks.file_id = files.file_id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    scored = []
    for r in rows:
        path, content, blob = r[0], r[1], r[2]
        vec = blob_to_vec(blob)
        score = embedder.cosine_sim(q_vec, vec)
        scored.append((path, content, 1.0 - float(score)))
    scored.sort(key=lambda x: x[2])
    return scored[:top_k]

def ask_llama(query, results):
    # Build context from retrieved chunks
    context = "\n\n".join([c for _, c, _ in results])
    prompt = f"""
    You are a helpful assistant.
    Answer the question based only on the context below:

    Context:
    {context}

    Question: {query}

    Answer:
    """
    t0 = time.perf_counter()
    resp = ollama.generate(model="llama3.2:3b", prompt=prompt)
    _log.info("llm answer | elapsed=%.3fs len=%s", time.perf_counter() - t0, len(resp.get("response", "")))
    return resp["response"]

if __name__ == "__main__":
    query = "How does embedding work in this project?"
    results = search_db(query, top_k=5)

    print("🔎 Retrieved chunks:")
    for path, content, dist in results:
        print(f"{path} | Score={1-dist:.3f}\n{content[:200]}...\n")

    print("🤖 Llama answer:")
    print(ask_llama(query, results))
