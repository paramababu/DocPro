from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import DEDUP_METHOD, DEDUP_SIM_THRESHOLD
from . import embedder


def _normalize_text(s: str) -> str:
    # Lowercase, strip, collapse whitespace
    return " ".join(s.lower().strip().split())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0.0 else 0.0


def deduplicate_chunks(
    chunks: List[str],
    method: str | None = None,
    sim_threshold: float | None = None,
) -> List[str]:
    """
    Remove duplicate or near-duplicate chunks.

    - method='none'      => no deduplication
    - method='exact'     => drop exact-normalized duplicates
    - method='semantic'  => drop chunks whose embedding similarity to any kept
                             chunk is >= sim_threshold
    """
    m = (method or DEDUP_METHOD).lower()
    if m == "none" or not chunks:
        return chunks

    if m == "exact":
        seen = set()
        out: List[str] = []
        for c in chunks:
            key = _normalize_text(c)
            if key and key not in seen:
                seen.add(key)
                out.append(c)
        return out

    if m == "semantic":
        thr = sim_threshold if sim_threshold is not None else DEDUP_SIM_THRESHOLD
        vecs = embedder.embed_texts(chunks)
        kept_indices: List[int] = []
        kept_vecs: List[np.ndarray] = []
        for i, v in enumerate(vecs):
            is_dup = False
            for kv in kept_vecs:
                if _cosine(kv, v) >= thr:
                    is_dup = True
                    break
            if not is_dup:
                kept_indices.append(i)
                kept_vecs.append(v)
        return [chunks[i] for i in kept_indices]

    return chunks

