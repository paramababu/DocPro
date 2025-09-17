from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple

from .db import get_conn, ensure_schema
from . import config


def list_repos() -> List[Tuple[int, str, str, datetime]]:
    ensure_schema(config.PG_EMBED_DIM)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT repo_id, url, branch, created_at FROM repos ORDER BY created_at DESC"
            )
            rows = cur.fetchall()
    return [(int(r[0]), str(r[1]), str(r[2]), r[3]) for r in rows]


def prune_session_repos(prefix: str = "vectorstore://", older_than_hours: int = 24) -> int:
    """Delete session repos older than the given hours.

    Returns number of repos deleted.
    """
    ensure_schema(config.PG_EMBED_DIM)
    cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM repos WHERE url LIKE %s AND created_at < %s RETURNING repo_id",
                (prefix + "%", cutoff),
            )
            deleted = cur.fetchall()
        conn.commit()
    return len(deleted)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Small pgvector utilities")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List repos")

    p = sub.add_parser("prune", help="Prune session repos older than N hours")
    p.add_argument("--prefix", default="vectorstore://")
    p.add_argument("--hours", type=int, default=24)

    args = ap.parse_args()
    if args.cmd == "list":
        for rid, url, branch, ts in list_repos():
            print(f"{rid}\t{url}\t{branch}\t{ts}")
    elif args.cmd == "prune":
        n = prune_session_repos(prefix=args.prefix, older_than_hours=args.hours)
        print(f"Deleted {n} repo(s)")
    else:
        ap.print_help()

