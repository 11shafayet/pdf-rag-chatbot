# PDF RAG Chatbot

Upload a PDF and ask questions about it. The chatbot answers using RAG (Retrieval-Augmented Generation), so every answer is grounded in the actual document, with page-level citations and a confidence score.

## How it works

- Parses PDFs with PyMuPDF and splits them into chunks
- Embeds chunks with sentence-transformers and indexes them in FAISS
- Also runs BM25 keyword search alongside the vector search, then fuses both scores for better retrieval
- Re-ranks the top matches before sending them to the LLM (via Groq)
- Shows the answer with a confidence label (🟢 High / 🟡 Medium / 🔴 Low) and the exact source, page number, and chunk used

## Tech stack

Python, Streamlit, FAISS, sentence-transformers, rank-bm25, PyMuPDF, Groq

## Setup

1. Clone the repo
```bash
git clone https://github.com/11shafayet/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add your Groq API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

4. Run the app
```bash
streamlit run app.py
```

Then open the local URL Streamlit gives you, upload a PDF, hit "Process PDFs", and start asking questions.

## Project structure

```
app.py            # Streamlit UI
src/               # RAG pipeline (ingestion, retrieval, reranking, generation)
data/indexes/      # Saved FAISS indexes
tests/             # Tests
```