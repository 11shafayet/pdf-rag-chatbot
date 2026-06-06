from sentence_transformers import CrossEncoder


class ReRanker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, question, chunks, top_k=3):
        if not chunks:
            return []

        pairs = [
            [question, chunk["text"]]
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        scored_chunks = []

        for chunk, score in zip(chunks, scores):
            chunk_copy = chunk.copy()
            chunk_copy["rerank_score"] = float(score)
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return scored_chunks[:top_k]