import os
import sys
import time
import logging
from pathlib import Path
from typing import Iterable

# Ensure project root (parent of this folder) is on sys.path
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from rag_app import loader, embedder, config
from rag_app.db import get_conn, ensure_schema


_log = logging.getLogger("docpro.ingest")
if not _log.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    _log.setLevel(getattr(logging, level, logging.INFO))


def _iter_project_files(project_path: str) -> Iterable[Path]:
    exts = {".md", ".py", ".txt", ".docx", ".doc", ".pdf"}
    for file in Path(project_path).rglob("*"):
        if file.is_file() and file.suffix.lower() in exts:
            yield file


def ingest_db(project_path: str) -> None:
    _log.info("ingest_db start | project_path=%s", project_path)
    ensure_schema(config.PG_EMBED_DIM)
    with get_conn() as conn:
        with conn.cursor() as cur:
            repo_url = str(Path(project_path).resolve())
            # Try to reuse existing repo row if present
            cur.execute("SELECT repo_id FROM repos WHERE url = %s AND branch = %s", (repo_url, "local"))
            row = cur.fetchone()
            if row:
                repo_id = row[0]
                _log.info("Using existing repo | repo_id=%s", repo_id)
            else:
                cur.execute(
                    "INSERT INTO repos (url, branch) VALUES (%s, %s) RETURNING repo_id",
                    (repo_url, "local"),
                )
                repo_id = cur.fetchone()[0]
                _log.info("Created repo | repo_id=%s", repo_id)

            for file in _iter_project_files(project_path):
                try:
                    text = loader.load_file(str(file))
                    _log.info("Loaded file | path=%s chars=%s", str(file), len(text))
                except Exception as e:
                    print(f"⚠️ Could not read {file}: {e}")
                    continue

                t_chunk = time.perf_counter()
                chunks = loader.chunk_text(text)
                _log.info("Chunked | path=%s chunks=%s elapsed=%.3fs", str(file), len(chunks), time.perf_counter() - t_chunk)
                if not chunks:
                    continue

                cur.execute(
                    "INSERT INTO files (repo_id, path, file_type) VALUES (%s, %s, %s) RETURNING file_id",
                    (repo_id, str(file), file.suffix),
                )
                file_id = cur.fetchone()[0]

                t_emb = time.perf_counter()
                vecs = embedder.embed_texts(chunks)
                _log.info("Embedded | path=%s dim=%s count=%s elapsed=%.3fs", str(file), len(vecs[0]) if len(vecs) else 0, len(vecs), time.perf_counter() - t_emb)
                for idx, (chunk, vec) in enumerate(zip(chunks, vecs)):
                    emb = [float(x) for x in vec.tolist()]
                    cur.execute(
                        "INSERT INTO chunks (file_id, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                        (file_id, idx, chunk, emb),
                    )
                _log.info("Inserted file | file_id=%s path=%s chunks=%s", file_id, str(file), len(chunks))
        conn.commit()
    _log.info("✅ Project ingested into Postgres with embeddings!")


if __name__ == "__main__":
    ingest_db("./")
