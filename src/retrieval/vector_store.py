import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer



class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

        self.index_path = "data/indexes/index.faiss"
        self.chunk_path = "data/indexes/chunks.pkl"

        self.load_index()

    def build_index(self, chunks):
        self.chunks = chunks

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        self.save_index()

    def save_index(self):
        """Save FAISS index and chunk metadata."""

        os.makedirs("data/indexes", exist_ok=True)

        faiss.write_index(
            self.index,
            self.index_path
        )

        with open(self.chunk_path, "wb") as file:
            pickle.dump(
                self.chunks,
                file
            )

    def load_index(self):
        """Load FAISS index and chunk metadata from disk."""

        if not os.path.exists(self.index_path):
            return False

        if not os.path.exists(self.chunk_path):
            return False

        self.index = faiss.read_index(self.index_path)

        with open(self.chunk_path, "rb") as file:
            self.chunks = pickle.load(file)

        return True

    def search(self, query, top_k=3):
        if self.index is None:
            return []

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for distance, index in zip(distances[0], indices[0]):
            if index == -1:
                continue

            chunk = self.chunks[index].copy()
            chunk["distance"] = float(distance)

            results.append(chunk)

        return results