import os
import tempfile

import streamlit as st

from src.pipeline.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("PDF RAG Chatbot")
st.write("Upload a PDF and ask source-grounded questions.")


if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = (
        st.session_state.rag.vector_store.index is not None
        and len(st.session_state.rag.vector_store.chunks) > 0
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Document Upload")

    if st.session_state.pdf_processed:
        st.success("Saved index loaded. You can ask questions.")
    else:
        st.info("No saved index loaded. Upload PDFs to begin.")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write("Selected files:")

        for uploaded_file in uploaded_files:
            st.write(f"- {uploaded_file.name}")

        if st.button("Process PDFs"):
            temp_pdf_paths = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_pdf_paths.append(temp_file.name)

            with st.spinner("Processing PDFs..."):
                stats = st.session_state.rag.ingest_pdfs(temp_pdf_paths)

            st.session_state.pdf_processed = True

            st.success("PDFs processed successfully.")
            st.write(f"Documents: {stats['documents']}")
            st.write(f"Pages: {stats['pages']}")
            st.write(f"Chunks: {stats['chunks']}")

            for temp_pdf_path in temp_pdf_paths:
                os.remove(temp_pdf_path)

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source']} — Page {source['page_number']} — Chunk {source['chunk_id']}**"
                    )
                    st.caption(f"Retrieval distance: {source['distance']:.4f}")

                    if source.get("bm25_score") is not None:
                        st.caption(f"BM25 score: {source['bm25_score']:.4f}")

                    if source.get("fusion_score") is not None:
                        st.caption(f"Fusion score: {source['fusion_score']:.4f}")

                    if source.get("rerank_score") is not None:
                        st.caption(f"Re-rank score: {source['rerank_score']:.4f}")

                    st.write(source["text_preview"])


if not st.session_state.pdf_processed:
    st.info("Upload and process a PDF first.")

question = st.chat_input("Ask a question about the PDF")

if question:
    if not st.session_state.pdf_processed:
        st.warning("Please upload and process a PDF before asking questions.")
    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag.answer_question(question)

            st.write(response["answer"])

            with st.expander("Sources"):
                for source in response["sources"]:
                    st.markdown(
                        f"**{source['source']} — Page {source['page_number']} — Chunk {source['chunk_id']}**"
                    )
                    st.caption(f"Retrieval distance: {source['distance']:.4f}")

                    if source.get("bm25_score") is not None:
                        st.caption(f"BM25 score: {source['bm25_score']:.4f}")

                    if source.get("fusion_score") is not None:
                        st.caption(f"Fusion score: {source['fusion_score']:.4f}")

                    if source.get("rerank_score") is not None:
                        st.caption(f"Re-rank score: {source['rerank_score']:.4f}")
    
                    st.write(source["text_preview"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"]
            }
        )