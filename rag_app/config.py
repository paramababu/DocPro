import os

LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
SEMANTIC_SIM_THRESHOLD = float(os.getenv("SEMANTIC_SIM_THRESHOLD", "0.75"))

CHUNKING_METHOD = os.getenv("CHUNKING_METHOD", "semantic").lower()

TOP_K = int(os.getenv("TOP_K", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
UPLOADS_PATH = "uploads"

USE_SPACY_SENTENCIZER = os.getenv("USE_SPACY_SENTENCIZER", "true").lower() in ("1", "true", "yes", "on")
SPACY_LANGUAGE = os.getenv("SPACY_LANGUAGE", "en")

DEDUP_METHOD = os.getenv("DEDUP_METHOD", "none").lower()  # 'none' | 'exact' | 'semantic'
DEDUP_SIM_THRESHOLD = float(os.getenv("DEDUP_SIM_THRESHOLD", "0.96"))
