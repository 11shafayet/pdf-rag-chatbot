from src.loaders.pdf_loader import load_pdf
from src.preprocessing.text_splitter import split_text
from src.llm.llm import LLMClient
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.reranker import ReRanker
from src.retrieval.bm25_store import BM25Store
from src.retrieval.vector_store import VectorStore
from src.retrieval.score_fusion import fuse_scores
from src.retrieval.confidence import calculate_confidence
from src.retrieval.citation import select_best_evidence_sentence
import time

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.reranker = ReRanker()
        self.chunks = []
        self.llm = LLMClient()
        self.bm25_store = BM25Store()
        self.query_rewriter = QueryRewriter()

    def ingest_pdfs(self, pdf_paths):
        start_total = time.time()

        all_pages = []

        start_loading = time.time()
        for pdf_path in pdf_paths:
            pages = load_pdf(pdf_path)
            all_pages.extend(pages)
        loading_time = time.time() - start_loading

        start_chunking = time.time()
        self.chunks = split_text(
            all_pages,
            chunk_size=1000,
            chunk_overlap=200
        )
        chunking_time = time.time() - start_chunking

        start_indexing = time.time()
        vector_stats = self.vector_store.build_index(self.chunks)
        self.bm25_store.build_index(self.chunks)
        indexing_time = time.time() - start_indexing

        total_time = time.time() - start_total

        return {
            "documents": len(pdf_paths),
            "pages": len(all_pages),
            "chunks": len(self.chunks),
            "loading_time": round(loading_time, 2),
            "chunking_time": round(chunking_time, 2),
            "indexing_time": round(indexing_time, 2),
            "total_time": round(total_time, 2),
            "embedding_time": vector_stats["embedding_time"],
            "save_time": vector_stats["save_time"],
        }
    
    def ingest_pdf(self, pdf_path):
        return self.ingest_pdfs([pdf_path])
        
    def retrieve(self, question, top_k=5):
        faiss_results = self.vector_store.search(
            query=question,
            top_k=20
        )

        bm25_results = self.bm25_store.search(
            query=question,
            top_k=20
        )

        merged_results = {}

        for chunk in faiss_results:
            key = (
                chunk["source"],
                chunk["page_number"],
                chunk["chunk_id"]
            )
            merged_results[key] = chunk

        for chunk in bm25_results:
            key = (
                chunk["source"],
                chunk["page_number"],
                chunk["chunk_id"]
            )

            if key in merged_results:
                merged_results[key]["bm25_score"] = chunk.get("bm25_score", 0)
            else:
                merged_results[key] = chunk

        candidate_chunks = list(merged_results.values())

        fused_chunks = fuse_scores(
            candidate_chunks,
            bm25_weight=0.4,
            faiss_weight=0.6
        )

        rerank_candidates = fused_chunks[:10]

        reranked_chunks = self.reranker.rerank(
            question=question,
            chunks=rerank_candidates,
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

    def answer_question(self, question, history=None):
        """receive user question, decide if retrieval is needed, search vector database if so, retrieve relevant chunks, build context, send context + question to LLM, return answer + sources."""
        if history is None:
            history = []

        rewrite_result = self.query_rewriter.rewrite(question, history)

        if not rewrite_result["needs_retrieval"]:
            return {
                "question": question,
                "answer": "I'm built to answer questions about your uploaded PDF — try asking something about the document.",
                "confidence": None,
                "sources": []
            }

        retrieved_chunks = self.retrieve(rewrite_result["search_query"])

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "I could not find this in the PDF.",
                "confidence": {"label": "Low", "score": 0.0},
                "sources": []
            }

        context = self.build_context(retrieved_chunks)

        answer = self.llm.generate_answer(
            question=question,
            context=context
        )

        confidence = calculate_confidence(
            chunks=retrieved_chunks,
            answer=answer
        )

        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources": [
                {
                    "source": chunk["source"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "bm25_score": chunk.get("bm25_score"),
                    "bm25_norm": chunk.get("bm25_norm"),
                    "faiss_norm": chunk.get("faiss_norm"),
                    "fusion_score": chunk.get("fusion_score"),
                    "rerank_score": chunk.get("rerank_score"),
                    "evidence_text": select_best_evidence_sentence(
                        question=question,
                        chunk_text=chunk["text"]
                    ),
                    "text_preview": chunk["text"][:250]
                }
                for chunk in retrieved_chunks
            ]
        }
     