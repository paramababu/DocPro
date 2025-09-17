from __future__ import annotations

import os
import logging
import psycopg2


_log = logging.getLogger("docpro.db")


def get_conn():
    """Create a PostgreSQL connection from environment variables.

    Env vars (with sensible defaults):
    - PGHOST (default: localhost)
    - PGPORT (default: 5432)
    - PGUSER (default: current OS user)
    - PGPASSWORD (default: empty)
    - PGDATABASE (default: docpro)
    """
    host = os.getenv("PGHOST", "localhost")
    port = int(os.getenv("PGPORT", "5432"))
    user = os.getenv("PGUSER") or os.getenv("USER") or os.getenv("USERNAME")
    dbname = os.getenv("PGDATABASE", "docpro")
    _log.debug("Connecting to Postgres | host=%s port=%s user=%s db=%s", host, port, user, dbname)
    return psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=os.getenv("PGPASSWORD", ""),
        dbname=dbname,
    )


def _get_existing_vector_dim(cur) -> int | None:
    _log.debug("Checking existing pgvector dimension for chunks.embedding")
    cur.execute(
        """
        SELECT a.atttypmod
        FROM pg_class c
        JOIN pg_attribute a ON a.attrelid = c.oid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = current_schema()
          AND c.relname = 'chunks'
          AND a.attname = 'embedding'
          AND a.atttypid::regtype::text = 'vector'
        """
    )
    row = cur.fetchone()
    if not row:
        _log.debug("No embedding column found yet")
        return None
    dim = int(row[0])
    _log.debug("Found embedding vector dimension=%s", dim)
    return dim if dim > 0 else None


def ensure_schema(embed_dim: int = 768) -> None:
    """Ensure pgvector extension and required tables exist.

    Note: If you change the embedding dimension, you must either drop and
    recreate the `chunks` table or migrate it accordingly.
    """
    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS repos (
        repo_id SERIAL PRIMARY KEY,
        url TEXT NOT NULL,
        branch TEXT DEFAULT 'main',
        commit_id TEXT,
        created_at TIMESTAMP DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS files (
        file_id SERIAL PRIMARY KEY,
        repo_id INT REFERENCES repos(repo_id) ON DELETE CASCADE,
        path TEXT NOT NULL,
        file_type TEXT,
        created_at TIMESTAMP DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id SERIAL PRIMARY KEY,
        file_id INT REFERENCES files(file_id) ON DELETE CASCADE,
        chunk_index INT,
        content TEXT NOT NULL,
        embedding vector({embed_dim}),
        created_at TIMESTAMP DEFAULT now()
    );
    """
    _log.info("Ensuring schema (pgvector + tables) | target_dim=%s", embed_dim)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            _log.debug("DDL executed")
            # Validate/align vector dimension if table exists
            dim = _get_existing_vector_dim(cur)
            if dim is not None and dim != embed_dim:
                _log.warning("Vector dim mismatch | existing=%s expected=%s", dim, embed_dim)
                cur.execute("SELECT COUNT(*) FROM chunks")
                count = int(cur.fetchone()[0])
                _log.debug("Existing chunks count=%s", count)
                if count == 0:
                    _log.info("Altering embedding column to vector(%s)", embed_dim)
                    cur.execute(
                        f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({embed_dim})"
                    )
                else:
                    _log.error("Cannot auto-alter embedding dim with existing data")
                    raise RuntimeError(
                        f"pgvector dimension mismatch: existing={dim}, expected={embed_dim}. "
                        "Drop and recreate tables, then re-index uploads."
                    )
        conn.commit()
    _log.info("Schema ready")
