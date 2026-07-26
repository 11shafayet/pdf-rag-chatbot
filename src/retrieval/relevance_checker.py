import requests

from src.llm.llm import LLMClient
from src.utils.json_parsing import parse_json_response


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
or provides necessary supporting context.

Being from the document is not enough on its own. A passage that only mentions the
paper's title, authors, affiliations, or publication details — or that consists of
copyright notices, licensing text, or page headers — is NOT relevant just because it
technically comes from the document. It must contain substantive content that actually
helps answer the specific question asked. Mark it irrelevant if it's off-topic, only
superficially shares a keyword with the question, or is boilerplate with no real bearing
on what's being asked.

Respond with ONLY a JSON object, no markdown fences, no explanation, in this exact shape,
with exactly one boolean per passage, in the same order they were given:
{"relevant": [true, false, true, ...]}"""

        passages_text = "\n\n".join(
            f"Passage {i + 1} (Page {chunk.get('page_number', '?')}):\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        )

        user_prompt = f"""Question:
{question}

Passages:
{passages_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            raw_response = self.llm.call(messages, temperature=0.0)
            parsed = parse_json_response(raw_response, expected_keys=["relevant"])
            flags = parsed["relevant"]

            if not isinstance(flags, list) or len(flags) != len(retrieved_chunks):
                raise ValueError("relevance flag count did not match chunk count")

            for chunk, flag in zip(retrieved_chunks, flags):
                chunk["relevant"] = bool(flag)

            return retrieved_chunks

        except (ValueError, KeyError, AttributeError, requests.exceptions.RequestException):
            for chunk in retrieved_chunks:
                chunk["relevant"] = True

            return retrieved_chunks