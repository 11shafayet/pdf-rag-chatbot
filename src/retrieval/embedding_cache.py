import hashlib
import os
import pickle


class EmbeddingCache:
    def __init__(self):
        self.cache_path = "data/indexes/embedding_cache.pkl"

        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as file:
                self.cache = pickle.load(file)
        else:
            self.cache = {}

    def get_key(self, text):
        return hashlib.md5(
            text.encode("utf-8")
        ).hexdigest()

    def get(self, text):
        key = self.get_key(text)
        return self.cache.get(key)

    def set(self, text, embedding):
        key = self.get_key(text)
        self.cache[key] = embedding

    def save(self):
        os.makedirs(
            "data/indexes",
            exist_ok=True
        )

        with open(self.cache_path, "wb") as file:
            pickle.dump(
                self.cache,
                file
            )