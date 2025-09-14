from pathlib import Path
import docx2txt
from pypdf import PdfReader
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n\n".join([p.extract_text() or "" for p in reader.pages])

def load_docx(path: str) -> str:
    return docx2txt.process(path) or ""

def load_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def load_file(path: str) -> str:
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        return load_pdf(path)
    elif ext in ("docx", "doc"):
        return load_docx(path)
    else:
        return load_txt(path)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        soft = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
        if soft != -1 and soft > start + 0.5 * chunk_size:
            end = soft + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks
