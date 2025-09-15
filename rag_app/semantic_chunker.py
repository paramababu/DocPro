from typing import List
import re

try:
    import spacy  # type: ignore
    _NLP = spacy.load("en_core_web_sm")
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False
    _NLP = None


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    # Prefer spaCy if available; otherwise use a simple regex fallback.
    if _HAS_SPACY and _NLP is not None:
        doc = _NLP(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences

    parts = re.split(r"(?<=[.!?]) +", text)
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences

def group_sentences(sentences: List[str], max_chunk_size: int = 1000, overlap: int = 50) -> List[str]:

    chunks = []
    current_chunk = []
    current_length = 0

    for s in sentences:
        sentence_length = len(s)

        if current_length + sentence_length <= max_chunk_size:
            current_chunk.append(s)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

                num_overlap = max(1, overlap // 10)
                overlap_sents = current_chunk[-num_overlap:] if num_overlap < len(current_chunk) else current_chunk

                current_chunk = overlap_sents + [s]
                current_length = sum(len(x) for x in current_chunk)
            else:
                chunks.append(s)
                current_chunk = []
                current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
