import streamlit as st
import os
from pathlib import Path
import ollama
import numpy as np
import docx2txt
from pypdf import PdfReader

LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n\n".join([p.extract_text() or "" for p in reader.pages])

def load_docx(path: str) -> str:
    return docx2txt.process(path) or ""

def load_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def load_file(path: str) -> str:
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        return load_pdf(path)
    elif ext in ("docx", "doc"):
        return load_docx(path)
    else:
        return load_txt(path)

def chunk_text(text: str, chunk_size: int, overlap: int):
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        soft = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
        if soft != -1 and soft > start + 0.5 * chunk_size:
            end = soft + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks

def embed_texts(texts):
    vectors = []
    for t in texts:
        try:
            res = ollama.embeddings(model=EMBED_MODEL, input=t)
        except TypeError:
            res = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        vec = res.get("embedding") or res.get("embeddings")
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)

def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

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

def ask_llama(prompt):
    resp = ollama.generate(model=LLAMA_MODEL, prompt=prompt, options={"temperature": TEMPERATURE})
    return resp.get("response", "").strip()

st.set_page_config(page_title="Mini RAG with Llama", layout="wide")
st.title(" Ask your Documents (Mini RAG)")

uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf","docx","doc","txt"], accept_multiple_files=True)

if uploaded_files:
    docs_text = []
    for uf in uploaded_files:
        save_path = Path("uploads") / uf.name
        Path("uploads").mkdir(exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
        docs_text.append(load_file(str(save_path)))

    all_text = "\n\n".join(docs_text)
    chunks = chunk_text(all_text, CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_vectors = embed_texts(chunks)

    st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    question = st.text_input("Ask a question about your documents:")
    if question:
        top_contexts = search(question, chunk_vectors, chunks, TOP_K)
        prompt = build_prompt(top_contexts, question)
        answer = ask_llama(prompt)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, (text, score) in enumerate(top_contexts, 1):
            with st.expander(f"Chunk {i} (score={score:.3f})"):
                st.write(text)
