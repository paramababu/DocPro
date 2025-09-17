import os
import sys
import time
import logging
import streamlit as st
import ollama

# Ensure project root (parent of this folder) is on sys.path
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from rag_app.db import get_conn, ensure_schema
from rag_app import embedder, config, loader


# Logging setup (controlled by LOG_LEVEL env; default INFO)
_log = logging.getLogger("docpro.app")
if not _log.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    _log.setLevel(getattr(logging, level, logging.INFO))
_log.info("Streamlit app starting")


def _snip(text: str, n: int = 160) -> str:
    return text if len(text) <= n else text[:n] + "…"

# Embedding backend (same as ingestion)

def search_db(query, top_k=5, *, repo_id: int | None = None):
    t0 = time.perf_counter()
    _log.info("search_db called | repo_id=%s top_k=%s qlen=%s", repo_id, top_k, len(query))
    conn = get_conn()
    cur = conn.cursor()

    _log.info("Embedding query…")
    q_emb = embedder.embed_texts([query])[0].tolist()
    _log.debug("Query embed dim=%s head=%s", len(q_emb), [float(x) for x in q_emb[:8]])
    vec_str = "[" + ",".join(str(float(x)) for x in q_emb) + "]"

    if repo_id is not None:
        cur.execute(
            """
            SELECT path, content, embedding <=> %s::vector AS distance
            FROM chunks
            JOIN files ON chunks.file_id = files.file_id
            WHERE files.repo_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vec_str, repo_id, vec_str, top_k),
        )
    else:
        # Fallback: search across all data (not typical for this POC)
        cur.execute(
            """
            SELECT path, content, embedding <=> %s::vector AS distance
            FROM chunks
            JOIN files ON chunks.file_id = files.file_id
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vec_str, vec_str, top_k),
        )
    results = cur.fetchall()
    for i, (path, content, dist) in enumerate(results):
        _log.debug("Result[%s] path=%s score=%.4f head=%r", i, path, 1 - float(dist), _snip(content))
    cur.close()
    conn.close()
    _log.info("search_db done | rows=%s elapsed=%.3fs", len(results), time.perf_counter() - t0)
    return results

def ask_llama(query, results):
    _log.info("ask_llama called | contexts=%s", len(results))
    _log.debug("Question=%r", query)
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
    _log.debug("Prompt head=%r", _snip(prompt, 400))
    resp = ollama.generate(model="llama3.2:3b", prompt=prompt)
    _log.info("ask_llama done | elapsed=%.3fs rlen=%s", time.perf_counter() - t0, len(resp.get("response", "")))
    _log.debug("Answer=%r", _snip(resp.get("response", ""), 400))
    return resp["response"]


def _ensure_session_repo() -> int:
    if "repo_id" in st.session_state and st.session_state["repo_id"]:
        return int(st.session_state["repo_id"])
    _log.info("Creating new session repo")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO repos (url, branch) VALUES (%s, %s) RETURNING repo_id",
                ("streamlit://session", "session"),
            )
            rid = int(cur.fetchone()[0])
        conn.commit()
    st.session_state["repo_id"] = rid
    _log.info("Session repo created | repo_id=%s", rid)
    return rid


def _delete_file_by_path(repo_id: int, path: str) -> None:
    # Remove an existing file (and cascaded chunks) for this repo, if present
    _log.debug("Deleting existing file if present | repo_id=%s path=%s", repo_id, path)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM files WHERE repo_id = %s AND path = %s",
                (int(repo_id), path),
            )
        conn.commit()


# (No DB admin tools in POC)


def index_uploaded_files(files) -> tuple[int, int, int]:
    repo_id = _ensure_session_repo()
    os.makedirs(config.UPLOADS_PATH, exist_ok=True)

    # If user clicks index repeatedly with same files, avoid duplicate inserts
    try:
        fps = tuple((uf.name, uf.size) for uf in files)  # type: ignore[attr-defined]
    except Exception:
        fps = tuple((uf.name, len(uf.getbuffer())) for uf in files)
    _log.info("index_uploaded_files called | repo_id=%s files=%s", repo_id, [n for n, _ in fps])
    if st.session_state.get("_indexed_fps") == fps:
        _log.info("No changes in uploaded files; skipping re-index")
        return repo_id, 0, 0

    file_count = 0
    chunk_count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for uf in files:
                save_path = os.path.join(config.UPLOADS_PATH, uf.name)
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
                _log.info("Saved upload | path=%s size=%s", save_path, os.path.getsize(save_path))
                text = loader.load_file(save_path)
                _log.info("Loaded text | path=%s chars=%s", save_path, len(text))
                # POC: simple character chunking without overlap to avoid duplicates
                t_chunk = time.perf_counter()
                chunks = loader.character_chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=0)
                _log.info("Chunked | path=%s chunks=%s elapsed=%.3fs", save_path, len(chunks), time.perf_counter() - t_chunk)
                for ci, ch in enumerate(chunks):
                    _log.debug("Chunk %s | len=%s | head=%r", ci, len(ch), _snip(ch))
                if not chunks:
                    continue
                # Idempotent per file: drop existing row for same path if exists
                _delete_file_by_path(repo_id, save_path)
                cur.execute(
                    "INSERT INTO files (repo_id, path, file_type) VALUES (%s, %s, %s) RETURNING file_id",
                    (repo_id, save_path, os.path.splitext(save_path)[1]),
                )
                file_id = int(cur.fetchone()[0])
                t_emb = time.perf_counter()
                vecs = embedder.embed_texts(chunks)
                if len(vecs):
                    _log.info("Embedded | path=%s dim=%s count=%s elapsed=%.3fs", save_path, len(vecs[0]), len(vecs), time.perf_counter() - t_emb)
                else:
                    _log.info("Embedded | path=%s dim=%s count=%s elapsed=%.3fs", save_path, 0, 0, time.perf_counter() - t_emb)
                for ci, (ch, vec) in enumerate(zip(chunks, vecs)):
                    _log.debug("Embed head for chunk %s | %s", ci, [float(x) for x in vec[:8].tolist()])
                for idx, (chunk, vec) in enumerate(zip(chunks, vecs)):
                    emb = [float(x) for x in vec.tolist()]
                    cur.execute(
                        "INSERT INTO chunks (file_id, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                        (file_id, idx, chunk, emb),
                    )
                    _log.debug("Inserted chunk | file_id=%s idx=%s len=%s", file_id, idx, len(chunk))
                _log.info("Inserted file | file_id=%s path=%s chunks=%s", file_id, save_path, len(chunks))
                file_count += 1
                chunk_count += len(chunks)
        conn.commit()
    _log.info("Indexing complete | repo_id=%s files=%s chunks=%s", repo_id, file_count, chunk_count)
    st.session_state["_indexed_fps"] = fps
    return repo_id, file_count, chunk_count

# Streamlit UI
st.set_page_config(page_title="DocPro RAG", layout="wide")
st.title(" Ask Your File (Postgres + Ollama)")

# Ensure schema (no admin UI in POC)
_log.info("Ensuring DB schema")
ensure_schema(config.PG_EMBED_DIM)

uploaded = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf","docx","doc","txt"], accept_multiple_files=True)
if uploaded:
    if st.button("Index uploaded file(s)"):
        with st.spinner("Indexing files into Postgres..."):
            try:
                rid, n_files, n_chunks = index_uploaded_files(uploaded)
            except Exception as e:
                st.error(f"Indexing failed: {e}")
                st.info("If this is a vector dimension mismatch, use 'Reset DB' in the sidebar and re-index.")
            else:
                if n_files == 0 and n_chunks == 0:
                    st.info("Files already indexed (no changes).")
                else:
                    st.success(f"Indexed {n_files} file(s), {n_chunks} chunk(s) into repo {rid}.")

query = st.text_input("Enter your question about the uploaded files:")

if st.button("Search"):
    if query.strip():
        with st.spinner("Searching your uploaded files..."):
            rid = st.session_state.get("repo_id")
            if not rid:
                st.warning("Please upload and index a file first.")
            else:
                try:
                    results = search_db(query, config.TOP_K, repo_id=int(rid))
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.info("If this is a vector dimension mismatch, recreate tables and re-index.")
                else:
                    # Answer first
                    st.subheader("🤖 LLaMA Answer")
                    answer = ask_llama(query, results)
                    st.write(answer)

                    # Then supporting chunks
                    st.subheader("🔎 Retrieved Chunks")
                    for path, content, dist in results:
                        st.markdown(f"**{path}** (Score={1-dist:.3f})")
                        st.code((content[:1000] + "...") if len(content) > 1000 else content)
