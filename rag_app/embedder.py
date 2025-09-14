import numpy as np
import ollama
from .config import EMBED_MODEL

def embed_texts(texts):
    vectors = []
    for t in texts:
        try:
            res = ollama.embeddings(model=EMBED_MODEL, input=t)
        except TypeError:
            res = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        vec = res.get("embedding") or res.get("embeddings")
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)

def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0
