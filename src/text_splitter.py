def split_text(pages,chunk_size=500,chunk_overlap=100):
    """
    Split PDF pages into overlapping chunks.
    """

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append(
                {
                    "page_number": page["page_number"],
                    "text": chunk_text
                }
            )

            start += chunk_size - chunk_overlap

    return chunks