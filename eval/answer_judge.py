import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from src.llm.llm import LLMClient
from src.utils.json_parsing import parse_json_response


class AnswerJudge:
    def __init__(self):
        self.llm = LLMClient()

    def judge(self, question, generated_answer, expected_answer, context, should_find_answer):
        """
        Returns:
        {
            "correct": bool | None,      # None when should_find_answer is False
            "faithful": bool | None,     # None when should_find_answer is False
            "refusal_correct": bool | None  # only meaningful when should_find_answer is False
        }
        Fails open on error: correct/faithful default to True, refusal_correct to False,
        the least-alarming assumption per axis rather than a blanket guess.
        """

        system_prompt = """You are an evaluation judge for a PDF question-answering system.

        You will be given a question, the system's generated answer, a reference expected answer,
        and the retrieved context passages the system had access to when generating its answer.

        Judge three things:

        1. correct: does the generated answer convey the same substantive information as the
        expected answer? Minor wording differences are fine; missing or wrong facts are not.

        2. faithful: is every claim in the generated answer actually supported by the retrieved
        context? An answer can be factually correct in general knowledge but still unfaithful
        if it states something the given context doesn't actually support — that counts as
        a failure here, since this system must only answer from the provided context.

        3. refusal_correct: only relevant if the question has no real answer in the document.
        If so, did the system correctly decline to answer (e.g. "I could not find this in
        the PDF") rather than inventing an answer?

        Respond with ONLY a JSON object, no markdown fences, no explanation, in this exact shape:
        {"correct": true, "faithful": true, "refusal_correct": null}
        """

        user_prompt = f"""Question:
            {question}

            Generated answer:
            {generated_answer}

            Expected answer:
            {expected_answer}

            Retrieved context:
            {context}

            Should this question have a real answer in the document? {should_find_answer}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            raw_response = self.llm.call(messages)
            parsed = parse_json_response(raw_response, expected_keys=["correct", "faithful", "refusal_correct"])

            if should_find_answer:
                return {
                    "correct": bool(parsed["correct"]),
                    "faithful": bool(parsed["faithful"]),
                    "refusal_correct": None
                }
            else:
                return {
                    "correct": None,
                    "faithful": None,
                    "refusal_correct": bool(parsed["refusal_correct"])
                }

        except (ValueError, KeyError, AttributeError, requests.exceptions.RequestException):
            if should_find_answer:
                return {"correct": True, "faithful": True, "refusal_correct": None}
            else:
                return {"correct": None, "faithful": None, "refusal_correct": False}