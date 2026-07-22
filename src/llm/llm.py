import os
import requests
from dotenv import load_dotenv
load_dotenv()

class LLMClient:
    def __init__(self):
        self.model = "openai/gpt-oss-20b"
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = os.environ.get("GROQ_API_KEY")

    def call(self, messages, temperature=0.2):
        """Send a messages list to Groq's chat completions endpoint, return the raw text response."""
        
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]



    def generate_answer(self, question, context):
        system_prompt = f"""
            You are a careful PDF question-answering assistant.

            Answer the user's question using ONLY the context below.

            Rules:
            - If the answer is not in the context, say: "I could not find this in the PDF."
            - Do not invent information.
            - Keep the answer clear and concise.
            - Mention the relevant page number if available.
        """
        user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
        messages= [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
            
        return self.call(messages)