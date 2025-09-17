from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .. import embedder


class MemoryStore:
    """Simple in-memory vector store using existing embedder.

    Keeps this project mid-level and easy to reason about.
    """

    def __init__(self) -> None:
        self._chunks: List[str] = []
        self._vectors: np.ndarray | None = None

    def reset(self) -> None:
        self._chunks = []
        self._vectors = None

    def index_texts(self, texts: List[str]) -> None:
        self._chunks = list(texts)
        self._vectors = embedder.embed_texts(self._chunks)

    def similarity_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        assert self._vectors is not None, "Vector store not indexed"
        q = embedder.embed_texts([query])[0]
        scores: List[Tuple[str, float]] = []
        for i, cvec in enumerate(self._vectors):
            s = embedder.cosine_sim(q, cvec)
            scores.append((self._chunks[i], float(s)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def get_store() -> MemoryStore:
    return MemoryStore()
