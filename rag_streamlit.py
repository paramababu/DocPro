import streamlit as st
from pathlib import Path
from rag_app import loader, retriever, embedder, llm, config

st.set_page_config(page_title="Mini RAG with Llama", layout="wide")
st.title("Ask your Documents (Mini RAG)")

# Prepare session state
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "vectors" not in st.session_state:
    st.session_state["vectors"] = []
if "uploaded_names" not in st.session_state:
    st.session_state["uploaded_names"] = []
if "index_done" not in st.session_state:
    st.session_state["index_done"] = False

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT",
    type=["pdf", "docx", "doc", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    Path(config.UPLOADS_PATH).mkdir(exist_ok=True)

    # Re-index only if the selection changed
    current_names = sorted([uf.name for uf in uploaded_files])
    selection_changed = current_names != st.session_state["uploaded_names"]

    if selection_changed:
        st.session_state["uploaded_names"] = current_names
        st.session_state["chunks"] = []
        st.session_state["vectors"] = []
        st.session_state["index_done"] = False

        status = st.empty()
        try:
            processed = 0
            max_chunks = getattr(config, "MAX_INDEX_CHUNKS", 2000)
            for uf in uploaded_files:
                save_path = Path(config.UPLOADS_PATH) / uf.name
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())

                # Stream chunks to keep memory usage stable
                for dc in loader.file_to_chunks(str(save_path), source_id=uf.name):
                    if processed >= max_chunks:
                        st.warning(
                            f"Reached MAX_INDEX_CHUNKS={max_chunks}. Stopping further indexing."
                        )
                        break
                    st.session_state["chunks"].append(dc.text)
                    # Embed per chunk to avoid large arrays
                    vec = embedder.embed_texts([dc.text])[0]
                    st.session_state["vectors"].append(vec)
                    processed += 1
                    if processed % 25 == 0:
                        status.write(f"Indexed {processed} chunks...")
                if processed >= max_chunks:
                    break

            st.session_state["index_done"] = processed > 0
            status.write(f"Indexing complete. Total chunks: {processed}")
        except MemoryError:
            st.error(
                "Ran out of memory while indexing. Try reducing CHUNK_SIZE/CHUNK_OVERLAP, "
                "or upload fewer/lighter files."
            )
        except Exception as e:
            st.error(f"Error during indexing: {e}")

    # Once indexed, enable Q&A
    if st.session_state.get("index_done"):
        st.success(
            f"Indexed {len(st.session_state['chunks'])} chunks from {len(uploaded_files)} file(s)."
        )
        question = st.text_input("Ask a question about your documents:")
        if question:
            vectors = st.session_state["vectors"]
            chunks = st.session_state["chunks"]
            top_contexts = retriever.search(question, vectors, chunks, config.TOP_K)
            prompt = retriever.build_prompt(top_contexts, question)
            answer = llm.ask_llama(prompt)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for i, (text, score) in enumerate(top_contexts, 1):
                with st.expander(f"Chunk {i} (score={score:.3f})"):
                    st.write(text)

