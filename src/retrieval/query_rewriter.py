from src.llm.llm import LLMClient

import json
import requests

class QueryRewriter:
    def __init__(self):
        self.llm = LLMClient()

    def rewrite(self, question, history):
        """
        history: list of {"role": ..., "content": ...} dicts, trimmed to last 10.
        Returns something like {"needs_retrieval": bool, "search_query": str}
        """
        system_prompt = """You are a query rewriting assistant for a PDF question-answering system.

            Given a conversation history and a new user question, decide two things:

            1. needs_retrieval: does answering this question require searching the PDF again?
            - True for questions about the document's content, even follow-ups like "what about X" or "explain that more."
            - False only for pure chit-chat, greetings, or meta-questions that don't need document content (e.g. "thanks", "hi", "what can you help with").

            2. search_query: a standalone, self-contained version of the question, rewritten so it
            makes sense with NO prior context. Resolve pronouns and vague references using the
            history (e.g. "what about the decoder?" after a question about encoders becomes
            "What does the decoder do in the Transformer architecture?").
            If needs_retrieval is False, just return the original question unchanged here.

            Respond with ONLY a JSON object, no markdown fences, no explanation, in this exact shape:
            {"needs_retrieval": true, "search_query": "..."}
        """

        history_text = "\n".join(
            f"{turn['role']}: {turn['content']}"
            for turn in history
        )

        user_prompt = f"""
            Conversation history:
            {history_text}

            New question:
            {question}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        raw_response = self.llm.call(messages)

        try:
            cleaned = raw_response.strip()

            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)

            return {
                "needs_retrieval": bool(parsed.get("needs_retrieval", True)),
                "search_query": parsed.get("search_query") or question
            }

        except (json.JSONDecodeError, AttributeError, requests.exceptions.RequestException): 
            return { 
                "needs_retrieval": True, 
                "search_query": question 
            }