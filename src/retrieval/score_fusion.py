def normalize_scores(items, score_key, output_key, reverse=False):
    """
    Normalize scores between 0 and 1.

    reverse=True means lower score is better.
    Example: FAISS distance.
    """

    valid_items = [
        item for item in items
        if item.get(score_key) is not None
    ]

    if not valid_items:
        return items

    scores = [item[score_key] for item in valid_items]

    min_score = min(scores)
    max_score = max(scores)

    for item in valid_items:
        if max_score == min_score:
            item[output_key] = 1.0
        else:
            normalized = (item[score_key] - min_score) / (max_score - min_score)

            if reverse:
                normalized = 1 - normalized

            item[output_key] = normalized

    return items


def fuse_scores(chunks, bm25_weight=0.4, faiss_weight=0.6):
    """
    Combine BM25 and FAISS scores into one hybrid score.
    """

    chunks = normalize_scores(
        chunks,
        score_key="bm25_score",
        output_key="bm25_norm",
        reverse=False
    )

    chunks = normalize_scores(
        chunks,
        score_key="distance",
        output_key="faiss_norm",
        reverse=True
    )

    for chunk in chunks:
        bm25_score = chunk.get("bm25_norm", 0)
        faiss_score = chunk.get("faiss_norm", 0)

        chunk["fusion_score"] = (
            bm25_weight * bm25_score
            + faiss_weight * faiss_score
        )

    chunks.sort(
        key=lambda chunk: chunk["fusion_score"],
        reverse=True
    )

    return chunks