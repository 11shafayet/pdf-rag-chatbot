from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()

stats = rag.ingest_pdf("data/uploads/sample.pdf")
print("PDF processed:")
print(stats)

question = input("\nAsk a question about the PDF: ")

response = rag.answer_question(question)

print("\nAnswer:")
print(response["answer"])

print("\nSources:")
for source in response["sources"]:
    print(f"- Page {source['page_number']}: {source['text_preview']}")