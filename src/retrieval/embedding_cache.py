import hashlib


class EmbeddingCache:
    def __init__(self):
        self.cache = {}

    def get_key(self, text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text):
        key = self.get_key(text)
        return self.cache.get(key)

    def set(self, text, embedding):
        key = self.get_key(text)
        self.cache[key] = embedding