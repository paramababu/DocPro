"""RAG application package modules.

Explicitly expose common submodules so that callers can do:
    from rag_app import loader, embedder, config, dedup, db
"""

from . import loader as loader  
from . import embedder as embedder  
from . import config as config  
from . import dedup as dedup  
from . import db as db  

__all__ = [
    "loader",
    "embedder",
    "config",
    "dedup",
    "db",
]
