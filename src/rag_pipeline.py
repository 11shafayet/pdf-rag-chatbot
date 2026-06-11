from src.pdf_loader import load_pdf
from src.text_splitter import split_text
from src.vector_store import VectorStore
from src.llm import LLMClient
from src.reranker import ReRanker

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.reranker = ReRanker()
        self.chunks = []
        self.llm = LLMClient()

    def ingest_pdfs(self, pdf_paths):
        all_pages = []

        for pdf_path in pdf_paths:
            pages = load_pdf(pdf_path)
            all_pages.extend(pages)

        self.chunks = split_text(
            all_pages,
            chunk_size=1000,
            chunk_overlap=200
        )

        self.vector_store.build_index(self.chunks)

        return {
            "documents": len(pdf_paths),
            "pages": len(all_pages),
            "chunks": len(self.chunks)
        }
    
    def ingest_pdf(self, pdf_path):
        return self.ingest_pdfs([pdf_path])
        
    def retrieve(self, question, top_k=5):
        candidate_chunks = self.vector_store.search(
            query=question,
            top_k=20
        )

        reranked_chunks = self.reranker.rerank(
            question=question,
            chunks=candidate_chunks,
            top_k=top_k
        )

        return reranked_chunks

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

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "I could not find this in the PDF.",
                "sources": []
            }

        best_distance = retrieved_chunks[0]["distance"]

        DISTANCE_THRESHOLD = 1.55

        if best_distance > DISTANCE_THRESHOLD:
            return {
                "question": question,
                "answer": "I could not find this in the PDF.",
                "sources": [
                    {
                        "source": chunk["source"],
                        "page_number": chunk["page_number"],
                        "chunk_id": chunk["chunk_id"],
                        "distance": chunk["distance"],
                        "rerank_score": chunk.get("rerank_score"),
                        "text_preview": chunk["text"][:250]
                    }
                    for chunk in retrieved_chunks
                ]
            }

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
                    "source": chunk["source"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "distance": chunk["distance"],
                    "rerank_score": chunk.get("rerank_score"),
                    "text_preview": chunk["text"][:250]
                }
                for chunk in retrieved_chunks
            ]
        }
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
                    "source": chunk["source"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "distance": chunk["distance"],
                    "text_preview": chunk["text"][:250]
                }
                for chunk in retrieved_chunks
            ]
        }