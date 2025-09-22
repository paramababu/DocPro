from __future__ import annotations

import logging

import numpy as np
import psycopg

from . import config


_log = logging.getLogger("docpro.db")


def get_conn():
    """Create a PostgreSQL connection using the configured DSN."""

    dsn = config.DB_DSN
    _log.debug("Connecting to PostgreSQL | dsn=%s", dsn)
    conn = psycopg.connect(dsn, autocommit=False)
    return conn


def vec_to_blob(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def ensure_schema(embed_dim: int = 0) -> None:
    """Ensure PostgreSQL tables exist (repos/files/chunks)."""

    _log.info("Ensuring PostgreSQL schema")
    statements = [
        """
        CREATE TABLE IF NOT EXISTS repos (
            repo_id SERIAL PRIMARY KEY,
            url TEXT NOT NULL,
            branch TEXT DEFAULT 'main',
            commit_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS files (
            file_id SERIAL PRIMARY KEY,
            repo_id INT REFERENCES repos(repo_id) ON DELETE CASCADE,
            path TEXT NOT NULL,
            file_type TEXT,
            commit_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id SERIAL PRIMARY KEY,
            file_id INT REFERENCES files(file_id) ON DELETE CASCADE,
            chunk_index INT,
            content TEXT NOT NULL,
            embedding BYTEA,
            repo_id INT,
            file_path TEXT,
            content_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS files_repo_id_idx ON files (repo_id)",
        "CREATE INDEX IF NOT EXISTS chunks_file_idx ON chunks (file_id, chunk_index)",
        "CREATE INDEX IF NOT EXISTS chunks_repo_id_idx ON chunks (repo_id)",
        "CREATE INDEX IF NOT EXISTS chunks_file_path_idx ON chunks (file_path)",
        "CREATE UNIQUE INDEX IF NOT EXISTS chunks_repo_hash_unique ON chunks (repo_id, content_hash)",
    ]

    with get_conn() as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        conn.commit()
    _log.info("Schema ready (PostgreSQL)")
