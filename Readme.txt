For virtual env

    # Windows (PowerShell)
    python -m venv .venv
    .venv\\Scripts\\Activate.ps1

    # macOS/Linux
    python -m venv .venv
    source .venv/bin/activate

install lib
    pip install --upgrade pip
    pip install -r requirements.txt

run ollama
    ollama serve
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text

SQLite setup
    # Default DB path: rag_app/../docpro.sqlite3 (config.DB_PATH)
    # No DB install needed. Schema is auto-created.

Run UI (SQLite-backed):
    streamlit run rag_app/streamlit_app.py

Maintenance utilities
    # List repos stored in SQLite
    python -m rag_app.pg_utils list

    # Prune session repos older than 24 hours (default)
    python -m rag_app.pg_utils prune --hours 24
    
