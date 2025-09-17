import ollama

# Quick test
resp = ollama.generate(
    model="llama3.2:3b",   
    prompt="Hello Ollama, are you running?"
)

print("✅ Ollama response:\n")
print(resp["response"])
