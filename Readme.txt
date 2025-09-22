For virtual env (use Python 3.12 recommended)

    # Windows (PowerShell)
    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # macOS/Linux
    python3.12 -m venv .venv
    source .venv/bin/activate

install lib
    # Upgrade tooling and install
    pip install -U pip setuptools wheel
    pip install -r requirements.txt

run ollama
    ollama serve
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text

PostgreSQL setup
    # Ensure PostgreSQL 14+ is running and a database exists, e.g.:
    #   createdb docpro
    # Configure connection string via DB_DSN (default: postgresql://localhost/docpro)
    #   export DB_DSN=postgresql://user:password@localhost:5432/docpro
    # Schema is created automatically on first run.

Run UI (PostgreSQL-backed):
    streamlit run rag_app/streamlit_app.py


    
