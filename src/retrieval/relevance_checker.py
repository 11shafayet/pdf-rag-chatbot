import json
import requests

from src.llm.llm import LLMClient


class RelevanceChecker:
    def __init__(self):
        self.llm = LLMClient()

    def check(self, question, retrieved_chunks):
        """
        question: the user's question.
        retrieved_chunks: list of chunk dicts, as returned by RAGPipeline.retrieve().
        """

        if not retrieved_chunks:
            return retrieved_chunks

        system_prompt = """You are a relevance-judging assistant for a document question-answering system.

            You will be given a question and a numbered list of text passages retrieved from a document.

            For EACH passage, decide: does it contain information that helps answer the question?
            A passage counts as relevant if it directly answers the question, partially answers it,
            or provides necessary supporting context. Mark it irrelevant if it's off-topic, only
            superficially shares a keyword, or is boilerplate (like copyright notices, headers,
            page numbers) with no bearing on the question.

            Respond with ONLY a JSON object, no markdown fences, no explanation, in this exact shape,
            with exactly one boolean per passage, in the same order they were given:
            {"relevant": [true, false, true, ...]}
        """

        passages_text = "\n\n".join(
            f"Passage {i + 1} (Page {chunk.get('page_number', '?')}):\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        )

        user_prompt = f"""Question:
            {question}

            Passages:
            {passages_text}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            raw_response = self.llm.call(messages)

            cleaned = raw_response.strip()

            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            flags = parsed["relevant"]

            if not isinstance(flags, list) or len(flags) != len(retrieved_chunks):
                raise ValueError("relevance flag count did not match chunk count")

            for chunk, flag in zip(retrieved_chunks, flags):
                chunk["relevant"] = bool(flag)

            return retrieved_chunks

        except (json.JSONDecodeError, AttributeError, KeyError, ValueError, requests.exceptions.RequestException):
            for chunk in retrieved_chunks:
                chunk["relevant"] = True

            return retrieved_chunks


    