import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content