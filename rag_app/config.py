import os

LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
UPLOADS_PATH = "uploads"
MAX_INDEX_CHUNKS = int(os.getenv("MAX_INDEX_CHUNKS", "2000"))
