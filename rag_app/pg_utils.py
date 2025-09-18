from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple, Optional

from .db import get_conn, ensure_schema
from . import config
from .pg_ingest import ingest_db


def list_repos() -> List[Tuple[int, str, str, datetime]]:
    ensure_schema()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT repo_id, url, branch, created_at FROM repos ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
        cur.close()
    return [(int(r[0]), str(r[1]), str(r[2]), r[3]) for r in rows]


def prune_session_repos(prefix: str = "vectorstore://", older_than_hours: int = 24) -> int:
    """Delete session repos older than the given hours.

    Returns number of repos deleted.
    """
    ensure_schema()
    cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
    with get_conn() as conn:
        cur = conn.cursor()
        cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')
        cur.execute(
            "DELETE FROM repos WHERE url LIKE ? AND created_at < ?",
            (prefix + "%", cutoff_str),
        )
        deleted = cur.rowcount
        conn.commit()
        cur.close()
    return max(deleted, 0)


def cleanup_repo(repo_id: Optional[int] = None, *, url: Optional[str] = None, branch: str = "local") -> int:
    """Delete a repo (and its files/chunks) by id or by (url, branch).

    Returns number of repos deleted (0 or 1).
    """
    ensure_schema()
    with get_conn() as conn:
        cur = conn.cursor()
            if repo_id is not None:
                cur.execute("SELECT 1 FROM repos WHERE repo_id = ?", (repo_id,))
                exists = cur.fetchone() is not None
                if exists:
                    cur.execute("DELETE FROM repos WHERE repo_id = ?", (repo_id,))
                    conn.commit()
                    cur.close()
                    return 1
                cur.close()
                return 0
            elif url is not None:
                cur.execute("SELECT repo_id FROM repos WHERE url = ? AND branch = ?", (url, branch))
                row = cur.fetchone()
                if row:
                    cur.execute("DELETE FROM repos WHERE url = ? AND branch = ?", (url, branch))
                    conn.commit()
                    cur.close()
                    return 1
                cur.close()
                return 0
            else:
                raise ValueError("Provide repo_id or url")


def reingest(project_path: str, *, cleanup: bool = True, branch: str = "local", commit_id: Optional[str] = None) -> None:
    """Cleanup existing repo by path+branch and ingest afresh.

    - If cleanup=True, deletes existing repo rows for (url=abs(project_path), branch).
    - Then calls ingest_db(project_path). commit_id is stored at repo-level if provided.
    """
    ensure_schema()
    from pathlib import Path

    repo_url = str(Path(project_path).resolve())
    with get_conn() as conn:
        cur = conn.cursor()
        if cleanup:
            cur.execute("DELETE FROM repos WHERE url = ? AND branch = ?", (repo_url, branch))
        if commit_id is not None:
            cur.execute(
                "INSERT INTO repos (url, branch, commit_id) VALUES (?, ?, ?)",
                (repo_url, branch, commit_id),
            )
            conn.commit()
        cur.close()
    ingest_db(project_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Small SQLite RAG utilities")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List repos")

    p = sub.add_parser("prune", help="Prune session repos older than N hours")
    p.add_argument("--prefix", default="vectorstore://")
    p.add_argument("--hours", type=int, default=24)

    c = sub.add_subparsers(dest="cmd2")

    cleanup_p = sub.add_parser("cleanup", help="Delete a repo by id or (url, branch)")
    cleanup_p.add_argument("--repo-id", type=int)
    cleanup_p.add_argument("--url")
    cleanup_p.add_argument("--branch", default="local")

    re_p = sub.add_parser("reingest", help="Cleanup and ingest a project path")
    re_p.add_argument("path")
    re_p.add_argument("--no-cleanup", action="store_true")
    re_p.add_argument("--branch", default="local")
    re_p.add_argument("--commit-id")

    args = ap.parse_args()
    if args.cmd == "list":
        for rid, url, branch, ts in list_repos():
            print(f"{rid}\t{url}\t{branch}\t{ts}")
    elif args.cmd == "prune":
        n = prune_session_repos(prefix=args.prefix, older_than_hours=args.hours)
        print(f"Deleted {n} repo(s)")
    elif args.cmd == "cleanup":
        n = cleanup_repo(repo_id=args.repo_id, url=args.url, branch=args.branch)
        print(f"Deleted {n} repo(s)")
    elif args.cmd == "reingest":
        reingest(args.path, cleanup=(not args.no_cleanup), branch=args.branch, commit_id=args.commit_id)
        print("Re-ingest complete")
    else:
        ap.print_help()
