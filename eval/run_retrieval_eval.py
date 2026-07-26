import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from src.pipeline.rag_pipeline import RAGPipeline

TOP_K = 5
PDF_PATH = "eval/attention.pdf"

with open("eval/ground_truth.json") as f:
    data = json.load(f)

pipeline = RAGPipeline()
pipeline.ingest_pdf(PDF_PATH)  

scored_results = []
unanswerable_results = []

for case in data["cases"]:
    retrieved = pipeline.retrieve(case["question"], top_k=TOP_K)
    retrieved_pages = [chunk["page_number"] for chunk in retrieved]

    if not case["should_find_answer"]:
        unanswerable_results.append({
            "id": case["id"],
            "question": case["question"],
            "retrieved_pages": retrieved_pages,
            "top_distance": retrieved[0]["distance"] if retrieved else None,
        })
        continue

    expected_pages = set(case["expected_page"])

    hit = any(p in expected_pages for p in retrieved_pages)

    on_target = sum(1 for p in retrieved_pages if p in expected_pages)
    precision = on_target / len(retrieved_pages) if retrieved_pages else 0.0

    reciprocal_rank = 0.0
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_pages:
            reciprocal_rank = 1 / rank
            break

    scored_results.append({
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "retrieved_pages": retrieved_pages,
        "expected_pages": sorted(expected_pages),
        "hit": hit,
        "precision": precision,
        "reciprocal_rank": reciprocal_rank,
    })

hit_rate = sum(r["hit"] for r in scored_results) / len(scored_results)
avg_precision = sum(r["precision"] for r in scored_results) / len(scored_results)
mrr = sum(r["reciprocal_rank"] for r in scored_results) / len(scored_results)

print(f"{'ID':8} {'Cat':12} {'Hit':6} {'Prec':6} {'RR':6}  Retrieved -> Expected")
for r in scored_results:
    print(
        f"{r['id']:8} {r['category']:12} {str(r['hit']):6} "
        f"{r['precision']:.2f}   {r['reciprocal_rank']:.2f}   "
        f"{r['retrieved_pages']} -> {r['expected_pages']}"
    )

print()
print(f"Hit Rate@{TOP_K}:   {hit_rate:.1%}")
print(f"Precision@{TOP_K}:  {avg_precision:.1%}")
print(f"MRR@{TOP_K}:        {mrr:.3f}")

print("\n--- Unanswerable / trap questions (inspect manually, not scored) ---")
for r in unanswerable_results:
    dist_str = f"{r['top_distance']:.4f}" if r["top_distance"] is not None else "n/a"
    print(f"{r['id']}: {r['question']}")
    print(f"  retrieved pages: {r['retrieved_pages']}  (top distance: {dist_str})")