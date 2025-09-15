from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Iterable, Iterator

import docx2txt
from pypdf import PdfReader

from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from .semantic_chunker import split_sentences, group_sentences

@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, str]


def load_text_from_pdf(path: str) -> str:
    logger = logging.getLogger("app")
    logger.info("Loading PDF: %s", path)
    t0 = __import__("time").perf_counter()
    try:
        reader = PdfReader(path)
    except Exception as e:
        logger.error("Error reading PDF %s: %s", path, e)
        raise

    total_pages = len(getattr(reader, "pages", []) or [])
    pages: List[str] = []

    for i, page in enumerate(reader.pages):
        pt0 = __import__("time").perf_counter()
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.error("Error extracting text from page %d of PDF %s: %s", i + 1, path, e)
            text = ""
        pages.append(text)
        pt_ms = (__import__("time").perf_counter() - pt0) * 1000
        if (i + 1) % 5 == 0:
            logger.info(
                "pdf.extract | file=%s | page=%d/%s | page_ms=%.1f",
                path,
                i + 1,
                (total_pages if total_pages else "?"),
                pt_ms,
            )

    total_ms = (__import__("time").perf_counter() - t0) * 1000
    logger.info(
        "pdf.extract done | file=%s | pages=%d/%s | chars=%d | ms=%.1f",
        path,
        len(pages),
        (total_pages if total_pages else "?"),
        sum(len(p) for p in pages),
        total_ms,
    )
    return "\n\n".join(pages)

def load_text_from_docx(path: str) -> str:
    try:
        return docx2txt.process(path) or ""
    except Exception:
        logging.getLogger("app").warning("docx2txt failed | file=%s", path)
        return ""
    
def load_text_from_file(path: str) -> str:
    logger = logging.getLogger("app")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        logger.info("load | file=%s | type=pdf", path)
        return load_text_from_pdf(path)
    if ext in (".docx", ".doc"):
        logger.info("load | file=%s | type=docx", path)
        return load_text_from_docx(path)
    logger.info("load | file=%s | type=text", path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_file(path: str) -> str:
    """Public helper matching rag_streamlit expectations."""
    return load_text_from_file(path)
    

def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks: List[str] = []
    start = 0
    n = len(text)
    logger = logging.getLogger("app")
    while start < n:
        end = min(start + chunk_size, n)
        soft = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
        if soft != -1 and soft > start + 0.5 * chunk_size:
            end = soft + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - chunk_overlap, 0)
        if start == 0 and end == n:
            break
        if start >= n:
            break
    logger.info(
        "chunk_text | chunk_size=%d | overlap=%d | total_chunks=%d | chars=%d",
        chunk_size,
        chunk_overlap,
        len(chunks),
        n,
    )
    return chunks


def _char_chunks_iter(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Iterator[str]:
    """Memory-efficient iterator version of chunk_text.

    Yields chunks without materializing the full list in memory.
    """
    if not text:
        return
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        soft = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
        if soft != -1 and soft > start + 0.5 * chunk_size:
            end = soft + 1
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        start = max(end - chunk_overlap, 0)
        if start == 0 and end == n:
            break
        if start >= n:
            break


def file_to_chunks(path: str, source_id: str) -> Iterable[DocumentChunk]:
    text = load_text_from_file(path)

    # Simple character-based chunking by default; switchable to semantic.
    mode = os.getenv("CHUNK_MODE", "char").lower()

    if mode == "semantic":
        # Note: semantic splitting currently materializes lists; for very large
        # files keep CHUNK_MODE=char to avoid high memory usage.
        sentences = split_sentences(text)
        parts_iter: Iterable[str] = group_sentences(
            sentences,
            max_chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
    else:
        parts_iter = _char_chunks_iter(
            text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    for i, part in enumerate(parts_iter):
        yield DocumentChunk(
            text=part,
            metadata={
                "source": os.path.basename(path),
                "source_path": os.path.abspath(path),
                "source_id": source_id,
                "chunk": str(i),
            },
        )

