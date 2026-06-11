from src.pipeline.rag_pipeline import RAGPipeline


TEST_CASES = [
    {
        "question": "What percentage of global electricity generation came from renewable sources in 2024?",
        "expected_page": 1
    },
    {
        "question": "Which renewable technology experienced the fastest growth?",
        "expected_page": 1
    },
    {
        "question": "Why have solar panels become more accessible?",
        "expected_page": 1
    },
    {
        "question": "How many jobs did the renewable energy sector create worldwide by 2024?",
        "expected_page": 2
    },
    {
        "question": "What happened in countries that invested early in clean energy infrastructure?",
        "expected_page": 2
    },
    {
        "question": "How much did global investment in renewable energy exceed in 2024?",
        "expected_page": 2
    },
    {
        "question": "Which type of energy project represented a significant portion of investments?",
        "expected_page": 2
    },
    {
        "question": "What are some major challenges facing renewable energy?",
        "expected_page": 3
    },
    {
        "question": "What technology is expected to help solve energy storage issues?",
        "expected_page": 3
    },
    {
        "question": "What percentage of global electricity generation could renewable energy provide by 2040?",
        "expected_page": 3
    },
]


def run_evaluation(pdf_path):
    rag = RAGPipeline()
    stats = rag.ingest_pdf(pdf_path)

    print("PDF processed:")
    print(stats)
    print()

    correct = 0

    for index, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected_page = test["expected_page"]

        retrieved_chunks = rag.retrieve(question, top_k=5)

        retrieved_pages = [
            chunk["page_number"]
            for chunk in retrieved_chunks
        ]

        is_correct = expected_page in retrieved_pages

        if is_correct:
            correct += 1

        print(f"Test {index}")
        print(f"Question: {question}")
        print(f"Expected Page: {expected_page}")
        print(f"Retrieved Pages: {retrieved_pages}")
        print(f"Correct: {is_correct}")
        print("-" * 50)

    accuracy = correct / len(TEST_CASES) * 100

    print()
    print(f"Retrieval Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    run_evaluation("data/uploads/sample.pdf")