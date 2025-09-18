from __future__ import annotations

import os
import logging
import sqlite3
from typing import Optional
import numpy as np
from . import config


_log = logging.getLogger("docpro.db")


def get_conn():
    """Create a SQLite connection to the local DB file.

    Uses config.DB_PATH (default: rag_app/../docpro.sqlite3). Ensures
    foreign keys are enabled and returns a connection usable across threads.
    """
    db_path = os.path.abspath(os.path.join(config.DB_PATH))
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _log.debug("Connecting to SQLite | path=%s", db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
    # Pragmas to improve concurrency and reduce locking issues
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except Exception:
        pass
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 10000;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def vec_to_blob(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def ensure_schema(embed_dim: int = 0) -> None:
    """Ensure SQLite tables exist (repos/files/chunks).

    Embeddings are stored as BLOB (float32 array). Adds helpful indexes.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS repos (
        repo_id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        branch TEXT DEFAULT 'main',
        commit_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS files (
        file_id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_id INT REFERENCES repos(repo_id) ON DELETE CASCADE,
        path TEXT NOT NULL,
        file_type TEXT,
        commit_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INT REFERENCES files(file_id) ON DELETE CASCADE,
        chunk_index INT,
        content TEXT NOT NULL,
        embedding BLOB,
        repo_id INT,
        file_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    _log.info("Ensuring SQLite schema")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.executescript(ddl)
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS files_repo_id_idx ON files (repo_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_file_idx ON chunks (file_id, chunk_index)")
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_repo_id_idx ON chunks (repo_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_file_path_idx ON chunks (file_path)")
            # Add content_hash for duplicate detection and a unique constraint per repo
            try:
                cur.execute("ALTER TABLE chunks ADD COLUMN content_hash TEXT")
            except Exception:
                pass
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS chunks_repo_hash_unique ON chunks (repo_id, content_hash)"
            )
        except Exception as e:
            _log.warning("Index create failed: %s", e)
        conn.commit()
    _log.info("Schema ready (SQLite)")
