import os
import pickle
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.retrieval.embedding_cache import EmbeddingCache


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

        self.index_path = "data/indexes/index.faiss"
        self.chunk_path = "data/indexes/chunks.pkl"

        self.embedding_cache = EmbeddingCache()

        self.load_index()

    def build_index(self, chunks):
        self.chunks = chunks

        texts = [chunk["text"] for chunk in chunks]

        start_embedding = time.time()

        embeddings = []

        texts_to_embed = []
        text_positions = []

        for position, text in enumerate(texts):
            cached_embedding = self.embedding_cache.get(text)

            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                texts_to_embed.append(text)
                text_positions.append(position)

        if texts_to_embed:
            new_embeddings = self.model.encode(
                texts_to_embed,
                batch_size=32,
                show_progress_bar=True
            )

            for text, position, embedding in zip(
                texts_to_embed,
                text_positions,
                new_embeddings
            ):
                self.embedding_cache.set(text, embedding)
                embeddings[position] = embedding

        embedding_time = time.time() - start_embedding

        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]

        start_faiss = time.time()

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        start_save = time.time()
        self.save_index()
        self.embedding_cache.save()
        save_time = time.time() - start_save

        return {
            "embedding_time": round(embedding_time, 2),
            "save_time": round(save_time, 2)
        }

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