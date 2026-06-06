import re


def split_into_sentences(text):
    """Split text into sentences using simple punctuation rules."""
    
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def split_text(
    pages,
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Split PDF pages into sentence-aware overlapping chunks.
    Preserve metadata for source citations.
    """

    chunks = []

    for page in pages:
        sentences = split_into_sentences(page["text"])

        current_chunk = ""
        chunk_id = 1

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                chunk_text = current_chunk.strip()

                if chunk_text:
                    chunks.append(
                        {
                            "source": page["source"],
                            "page_number": page["page_number"],
                            "chunk_id": chunk_id,
                            "text": chunk_text
                        }
                    )

                overlap_text = chunk_text[-chunk_overlap:]

                current_chunk = overlap_text + " " + sentence
                chunk_id += 1

        if current_chunk.strip():
            chunks.append(
                {
                    "source": page["source"],
                    "page_number": page["page_number"],
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip()
                }
            )

    return chunks