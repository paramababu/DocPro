"""RAG application package modules.

Explicitly expose common submodules so that callers can do:
    from rag_app import loader, embedder, config, dedup, db
"""

from . import loader as loader  # noqa: F401
from . import embedder as embedder  # noqa: F401
from . import config as config  # noqa: F401
from . import dedup as dedup  # noqa: F401
from . import db as db  # noqa: F401

__all__ = [
    "loader",
    "embedder",
    "config",
    "dedup",
    "db",
]
