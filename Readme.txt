For virtual env

    python -m venv .venv
    .venv\Scripts\activate   

install lib
    pip install --upgrade pip
    pip install -r requirements.txt

run ollama
    ollama serve
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text

Run UI:
    streamlit run rag_streamlit.py
    
