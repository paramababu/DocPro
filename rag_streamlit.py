import streamlit as st
from pathlib import Path
from rag_app import loader, retriever, embedder, llm, config
from rag_app.vectorstore import get_store
from rag_app.dedup import deduplicate_chunks

st.set_page_config(page_title="Mini RAG with Llama", layout="wide")
st.title("📄 Ask your Documents (Mini RAG)")

uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf","docx","doc","txt"], accept_multiple_files=True)

if uploaded_files:
    st.sidebar.header("Indexing Options")
    method = st.sidebar.selectbox(
        "Chunking method",
        options=["semantic", "character"],
        index=0 if config.CHUNKING_METHOD == "semantic" else 1,
    )
    dedup_method = st.sidebar.selectbox(
        "Duplicate detection",
        options=["none", "exact", "semantic"],
        index=["none", "exact", "semantic"].index(getattr(config, "DEDUP_METHOD", "none")),
    )
    if method == "semantic":
        token_target = st.sidebar.slider(
            "Target chunk size (tokens)",
            min_value=128,
            max_value=2048,
            value=config.CHUNK_SIZE_TOKENS,
            step=32,
        )
        token_overlap = st.sidebar.slider(
            "Overlap (tokens)",
            min_value=0,
            max_value=256,
            value=config.CHUNK_OVERLAP_TOKENS,
            step=8,
        )
        sim_threshold = st.sidebar.slider(
            "Semantic split threshold",
            min_value=0.5,
            max_value=0.95,
            value=float(config.SEMANTIC_SIM_THRESHOLD),
            step=0.01,
        )
        dedup_sim_threshold = st.sidebar.slider(
            "Duplicate sim threshold",
            min_value=0.8,
            max_value=0.999,
            value=float(getattr(config, "DEDUP_SIM_THRESHOLD", 0.96)),
            step=0.001,
        )
        char_chunk_size = config.CHUNK_SIZE
        char_overlap = config.CHUNK_OVERLAP
    else:
        char_chunk_size = st.sidebar.slider(
            "Target chunk size (chars)",
            min_value=256,
            max_value=4000,
            value=config.CHUNK_SIZE,
            step=64,
        )
        char_overlap = st.sidebar.slider(
            "Overlap (chars)",
            min_value=0,
            max_value=1000,
            value=config.CHUNK_OVERLAP,
            step=16,
        )
        token_target = config.CHUNK_SIZE_TOKENS
        token_overlap = config.CHUNK_OVERLAP_TOKENS
        sim_threshold = float(config.SEMANTIC_SIM_THRESHOLD)
        dedup_sim_threshold = float(getattr(config, "DEDUP_SIM_THRESHOLD", 0.96))
    docs_text = []
    Path(config.UPLOADS_PATH).mkdir(exist_ok=True)

    for uf in uploaded_files:
        save_path = Path(config.UPLOADS_PATH) / uf.name
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
        docs_text.append(loader.load_file(str(save_path)))

    all_text = "\n\n".join(docs_text)
    chunks = loader.chunk_text(
        all_text,
        chunk_size=char_chunk_size,
        overlap=char_overlap,
        method=method,
        token_target=token_target,
        token_overlap=token_overlap,
        sim_threshold=sim_threshold,
    )
    # Deduplicate chunks if requested
    if dedup_method != "none":
        before = len(chunks)
        chunks = deduplicate_chunks(chunks, method=dedup_method, sim_threshold=dedup_sim_threshold)
        st.info(f"Deduplicated chunks: {before} -> {len(chunks)}")

    # Index into vector store
    store = get_store()
    store.index_texts(chunks)

    st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    question = st.text_input("Ask a question about your documents:")
    if question:
        try:
            top_contexts = store.similarity_search(question, config.TOP_K)
        except Exception:
            # Fallback to in-memory cosine if store not available
            vecs = embedder.embed_texts(chunks)
            top_contexts = retriever.search(question, vecs, chunks, config.TOP_K)
        prompt = retriever.build_prompt(top_contexts, question)
        answer = llm.ask_llama(prompt)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, (text, score) in enumerate(top_contexts, 1):
            with st.expander(f"Chunk {i} (score={score:.3f})"):
                st.write(text)
