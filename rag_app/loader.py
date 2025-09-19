from __future__ import annotations

import os
import logging
from typing import List

from pypdf import PdfReader
import docx2txt

from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    SEMANTIC_SIM_THRESHOLD,
    CHUNKING_METHOD,
    USE_SPACY_SENTENCIZER,
    SPACY_LANGUAGE,
)
from . import embedder
import numpy as np


def load_pdf(path: str) -> str:
    logger = logging.getLogger("app")
    try:
        # Open the file explicitly to ensure the descriptor is closed promptly
        with open(path, "rb") as f:
            reader = PdfReader(f)
            return "\n\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        logger.warning("pdf.open failed | file=%s | err=%s", path, e)
        raise


def load_docx(path: str) -> str:
    try:
        return docx2txt.process(path) or ""
    except Exception:
        logging.getLogger("app").warning("docx2txt failed | file=%s", path)
        return ""


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in (".docx", ".doc"):
        return load_docx(path)
    return load_txt(path)


_SPACY_NLP = None


def _normalize_and_split_sentences(text: str) -> List[str]:
    import re
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraph_split = re.split(r"\n{2,}", text)
    sentences: List[str] = []

    if USE_SPACY_SENTENCIZER:
        try:
            global _SPACY_NLP
            if _SPACY_NLP is None:
                import spacy  # type: ignore
                _SPACY_NLP = spacy.blank(SPACY_LANGUAGE)
                if "sentencizer" not in _SPACY_NLP.pipe_names:
                    _SPACY_NLP.add_pipe("sentencizer")
            nlp = _SPACY_NLP
            for para in paragraph_split:
                para = para.strip()
                if not para:
                    continue
                doc = nlp(para)
                sentences.extend([s.text.strip() for s in doc.sents if s.text.strip()])
        except Exception:
            for para in paragraph_split:
                para = para.strip()
                if not para:
                    continue
                parts = re.split(r"(?<=[.!?])\s+", para)
                sentences.extend([p.strip() for p in parts if p.strip()])
    else:
        for para in paragraph_split:
            para = para.strip()
            if not para:
                continue
            parts = re.split(r"(?<=[.!?])\s+", para)
            sentences.extend([p.strip() for p in parts if p.strip()])
    if not sentences:
        sentences = [text]
    return sentences


def _estimate_tokens(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def _tail_overlap_by_tokens(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0 or not text.strip():
        return ""
    words = [w for w in text.strip().split() if w]
    tail = words[-overlap_tokens:]
    return " ".join(tail)


def character_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Sentence-aware chunking with character-based limits and soft overlaps."""
    if not text:
        return []

    sentences = _normalize_and_split_sentences(text)

    chunks: List[str] = []
    current = ""
    appended_since_finalize = False  # avoid emitting an overlap-only tail at the end

    def append_current():
        nonlocal current, appended_since_finalize
        if current.strip():
            chunks.append(current.strip())
        if overlap > 0 and current:
            tail = current[-overlap:]
            current = tail
            appended_since_finalize = False
        else:
            current = ""
            appended_since_finalize = False

    for sent in sentences:
        if not sent:
            continue

        if len(sent) > chunk_size:
            if current:
                append_current()
            start = 0
            while start < len(sent):
                piece = sent[start : start + chunk_size]
                if current and len(current) + len(piece) + 1 > chunk_size:
                    append_current()
                current = (current + (" " if current and not current.endswith(" ") else "") + piece)
                appended_since_finalize = True
                start += chunk_size
            continue

        # Normal sentence handling
        if not current:
            current = sent
            appended_since_finalize = True
        elif len(current) + 1 + len(sent) <= chunk_size:
            current += " " + sent
            appended_since_finalize = True
        else:
            append_current()
            current += ("" if not current else (" " if not current.endswith(" ") else "")) + sent
            appended_since_finalize = True

    # Emit leftover only if it contains new content (not just overlap tail)
    if current.strip() and (appended_since_finalize or not chunks):
        chunks.append(current.strip())

    return chunks


def semantic_chunk_text(
    text: str,
    target_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    sim_threshold: float = SEMANTIC_SIM_THRESHOLD,
) -> List[str]:
  
    if not text:
        return []

    sentences = _normalize_and_split_sentences(text)
    if not sentences:
        return []

    sent_vecs = embedder.embed_texts(sentences)

    chunks: List[str] = []
    current_text = ""
    current_count = 0
    centroid: np.ndarray | None = None
    count_vecs = 0

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom != 0.0 else 0.0

    def append_current():
        nonlocal current_text, current_count, centroid, count_vecs
        if current_text.strip():
            chunks.append(current_text.strip())
        overlap_prefix = _tail_overlap_by_tokens(current_text, overlap_tokens)
        current_text = overlap_prefix if overlap_prefix else ""
        current_count = _estimate_tokens(current_text)
        centroid = None
        count_vecs = 0

    for sent, vec in zip(sentences, sent_vecs):
        sent_tokens = _estimate_tokens(sent)
        if not current_text:
            current_text = sent
            current_count = sent_tokens
            centroid = np.array(vec, dtype=np.float32)
            count_vecs = 1
            continue

        # Similarity to current centroid
        sim = cosine(centroid, vec) if centroid is not None else 1.0
        would_exceed = (current_count + sent_tokens) > target_tokens
        low_sim = sim < sim_threshold and current_count >= max(32, target_tokens // 2)

        if would_exceed or low_sim:
            append_current()
            if current_text:
                current_text = (current_text + " " + sent).strip()
                current_count = _estimate_tokens(current_text)
            else:
                current_text = sent
                current_count = sent_tokens
            centroid = np.array(vec, dtype=np.float32)
            count_vecs = 1
        else:
            current_text = (current_text + (" " if not current_text.endswith(" ") else "") + sent).strip()
            current_count += sent_tokens
            if centroid is None:
                centroid = np.array(vec, dtype=np.float32)
                count_vecs = 1
            else:
                centroid = (centroid * count_vecs + vec) / (count_vecs + 1)
                count_vecs += 1

    if current_text.strip():
        chunks.append(current_text.strip())

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    *,
    method: str | None = None,
    token_target: int | None = None,
    token_overlap: int | None = None,
    sim_threshold: float | None = None,
) -> List[str]:
    """Dispatch to configured chunking method with optional runtime overrides.

    - When method == 'semantic' (or config default), uses `semantic_chunk_text`.
    - Otherwise, falls back to character-based sentence-aware chunking.
    """
    m = (method or CHUNKING_METHOD).lower()
    if m == "semantic":
        return semantic_chunk_text(
            text,
            target_tokens=token_target or CHUNK_SIZE_TOKENS,
            overlap_tokens=token_overlap or CHUNK_OVERLAP_TOKENS,
            sim_threshold=sim_threshold if sim_threshold is not None else SEMANTIC_SIM_THRESHOLD,
        )
    return character_chunk_text(text, chunk_size=chunk_size, overlap=overlap)


 
