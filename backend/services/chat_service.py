from src.pipeline.rag_pipeline import RAGPipeline


class ChatService:
    def __init__(self):
        self.rag = RAGPipeline()

    def process_pdf(self, file_path):
        return self.rag.ingest_pdfs([file_path])

    def chat(self, question):
        return self.rag.answer_question(question)