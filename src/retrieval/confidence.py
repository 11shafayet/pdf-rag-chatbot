def calculate_confidence(chunks, answer):
    """
    Calculate confidence from the top retrieved chunks.
    """

    if not chunks:
        return {
            "score": 0,
            "label": "Low"
        }

    answer_lower = answer.lower()

    not_found_phrases = [
        "i could not find this in the pdf",
        "not found in the pdf",
        "does not provide information",
        "does not mention",
        "not clearly present"
    ]

    if any(phrase in answer_lower for phrase in not_found_phrases):
        return {
            "score": 0,
            "label": "Low"
        }
    
    top_chunk = chunks[0]

    fusion = top_chunk.get("fusion_score", 0)
    rerank = top_chunk.get("rerank_score", 0)

    # Normalize rerank score to approximately 0-1
    rerank_norm = min(max(rerank / 10, 0), 1)

    confidence = (
        0.6 * fusion +
        0.4 * rerank_norm
    )

    confidence = round(confidence * 100, 2)

    if confidence >= 85:
        label = "High"
    elif confidence >= 60:
        label = "Medium"
    else:
        label = "Low"

    return {
        "score": confidence,
        "label": label
    }