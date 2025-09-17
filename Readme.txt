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

Postgres setup (pgvector)
    # Set env (or .env) as needed
    # PGHOST=localhost, PGPORT=5432, PGUSER=<you>, PGPASSWORD=, PGDATABASE=docpro
    # PG_EMBED_DIM must match your embed model (nomic-embed-text -> 768)

    # Ingest a project/repo into Postgres (creates tables if missing)
    python -m rag_app.pg_ingest  # defaults to current folder

Run UI (Postgres-backed only):
    streamlit run rag_app/streamlit_app.py

Maintenance utilities
    # List repos stored in Postgres
    python -m rag_app.pg_utils list

    # Prune session repos older than 24 hours (default)
    python -m rag_app.pg_utils prune --hours 24
    
