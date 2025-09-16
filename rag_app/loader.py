from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Iterable, List, Dict

from pypdf import PdfReader
import docx2txt

from .config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, str]


# ---- File loaders ----
def load_pdf(path: str) -> str:
    logger = logging.getLogger("app")
    try:
        reader = PdfReader(path)
    except Exception as e:
        logger.warning("pdf.open failed | file=%s | err=%s", path, e)
        raise
    return "\n\n".join([page.extract_text() or "" for page in reader.pages])


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


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Sentence-aware chunking with character-based limits and overlap.

    - Packs whole sentences up to `chunk_size` chars per chunk
    - Splits very long sentences into smaller slices
    - Adds `overlap` trailing characters from the previous chunk as prefix
      of the next chunk (not emitted as a final extra chunk)
    """
    if not text:
        return []

    import re

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split into sentences, also treating paragraph breaks as boundaries
    paragraph_split = re.split(r"\n{2,}", text)
    sentences: List[str] = []
    for para in paragraph_split:
        para = para.strip()
        if not para:
            continue
        parts = re.split(r"(?<=[.!?])\s+", para)
        sentences.extend([p.strip() for p in parts if p.strip()])

    # Fallback: if no sentence breaks were found, operate on raw text
    if not sentences:
        sentences = [text]

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

        # If a sentence is longer than the chunk size, slice it
        if len(sent) > chunk_size:
            # Close out any existing chunk before handling the giant sentence
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


def file_to_chunks(path: str, source_id: str) -> Iterable[DocumentChunk]:
    text = load_file(path)
    for i, part in enumerate(chunk_text(text)):
        yield DocumentChunk(
            text=part,
            metadata={
                "source": os.path.basename(path),
                "source_path": os.path.abspath(path),
                "source_id": source_id,
                "chunk": str(i),
            },
        )
