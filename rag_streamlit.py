import streamlit as st
from pathlib import Path
from rag_app import loader, retriever, embedder, llm, config

st.set_page_config(page_title="Mini RAG with Llama", layout="wide")
st.title("📄 Ask your Documents (Mini RAG)")

uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf","docx","doc","txt"], accept_multiple_files=True)

if uploaded_files:
    docs_text = []
    Path(config.UPLOADS_PATH).mkdir(exist_ok=True)

    for uf in uploaded_files:
        save_path = Path(config.UPLOADS_PATH) / uf.name
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
        docs_text.append(loader.load_file(str(save_path)))

    all_text = "\n\n".join(docs_text)
    chunks = loader.chunk_text(all_text)
    chunk_vectors = embedder.embed_texts(chunks)

    st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    question = st.text_input("Ask a question about your documents:")
    if question:
        top_contexts = retriever.search(question, chunk_vectors, chunks, config.TOP_K)
        prompt = retriever.build_prompt(top_contexts, question)
        answer = llm.ask_llama(prompt)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, (text, score) in enumerate(top_contexts, 1):
            with st.expander(f"Chunk {i} (score={score:.3f})"):
                st.write(text)
