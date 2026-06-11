import re
from rank_bm25 import BM25Okapi


class BM25Store:
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.tokenized_chunks = []

    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def build_index(self, chunks):
        self.chunks = chunks

        self.tokenized_chunks = [
            self.tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(self, query, top_k=10):
        if self.bm25 is None:
            return []

        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:
            chunk = self.chunks[index].copy()
            chunk["bm25_score"] = float(scores[index])
            results.append(chunk)

        return results