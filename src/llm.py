import requests


class LLMClient:
    def __init__(self):
        self.model = "llama3.2:3b"
        self.base_url = "http://localhost:11434/api/generate"

    def generate_answer(self, question, context):
        prompt = f"""
You are a careful PDF question-answering assistant.

Answer the user's question using ONLY the context below.

Rules:
- If the answer is not in the context, say: "I could not find this in the PDF."
- Do not invent information.
- Keep the answer clear and concise.
- Mention the relevant page number if available.

Context:
{context}

Question:
{question}

Answer:
"""

        response = requests.post(
            self.base_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json()["response"]