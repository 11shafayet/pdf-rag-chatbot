import re


def split_sentences(text):
    """
    Split text into sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

def select_best_evidence_sentence(question, chunk_text):
    """
    Select the best sentence from a chunk as evidence.

    Simple version:
    Score each sentence by keyword overlap with the question.
    """

    sentences = split_sentences(chunk_text)

    if not sentences:
        return chunk_text

    question_words = set(
        re.findall(r"\b\w+\b", question.lower())
    )

    best_sentence = sentences[0]
    best_score = 0

    for sentence in sentences:
        sentence_words = set(
            re.findall(r"\b\w+\b", sentence.lower())
        )

        overlap = question_words.intersection(sentence_words)
        score = len(overlap)

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence