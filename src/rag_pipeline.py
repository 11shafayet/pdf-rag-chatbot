from src.pdf_loader import load_pdf
from src.text_splitter import split_text
from src.vector_store import VectorStore
from src.llm import LLMClient

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.chunks = []
        self.llm = LLMClient()

    def ingest_pdf(self, pdf_path):
        """load a PDF file, split it into chunks, and build the vector index."""

        pages = load_pdf(pdf_path)

        self.chunks = split_text(
            pages,
            chunk_size=500,
            chunk_overlap=100
        )

        self.vector_store.build_index(self.chunks)

        return {
            "pages": len(pages),
            "chunks": len(self.chunks)
        }

    def retrieve(self, question, top_k=3):
        """retrieve relevant chunks from the vector store based on the question."""

        results = self.vector_store.search(question, top_k=top_k)
        return results

    def build_context(self, retrieved_chunks):
        """prepare chunks for the llm by formatting them with page numbers and text."""

        context_parts = []

        for chunk in retrieved_chunks:
            context_parts.append(
                f"[Page {chunk['page_number']}]\n{chunk['text']}"
            )

        return "\n\n".join(context_parts)

    def answer_question(self, question):
        """receive user question, search vector database, retrieve relevant chunks, build context, send context + question to LLM, return answer + sources."""
        
        retrieved_chunks = self.retrieve(question)

        context = self.build_context(retrieved_chunks)

        answer = self.llm.generate_answer(
            question=question,
            context=context
        )

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "page_number": chunk["page_number"],
                    "text_preview": chunk["text"][:200]
                }
                for chunk in retrieved_chunks
            ]
        }