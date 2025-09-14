import ollama
from .config import LLAMA_MODEL, TEMPERATURE

def ask_llama(prompt: str) -> str:
    resp = ollama.generate(model=LLAMA_MODEL, prompt=prompt, options={"temperature": TEMPERATURE})
    return resp.get("response", "").strip()
