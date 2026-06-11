from src.pipeline.rag_pipeline import RAGPipeline

rag = RAGPipeline()

test_files = [
    "data/uploads/sample.pdf",
    "data/uploads/sample.pdf"
]

stats = rag.ingest_pdfs(test_files)

print(stats)