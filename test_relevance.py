from src.retrieval.relevance_checker import RelevanceChecker

checker = RelevanceChecker()

# Exact rewritten query from the real pipeline run being compared against
QUESTION = "What architecture does the paper introduce?"

# Exact 5 chunk texts from that same run's Sources panel
CHUNKS = [
    {
        "page_number": 1,
        "text": "Provided proper attribution is provided, Google hereby grants permission to "
                "reproduce the tables and figures in this paper solely for use in journalistic "
                "or scholarly works."
    },
    {
        "page_number": 3,
        "text": "Figure 1: The Transformer - model architecture. The Transformer follows this "
                "overall architecture using stacked self-attention and point-wise, fully "
                "connected layers for both the encoder and decoder, shown in the left and "
                "right halves of Figure 1, re"
    },
    {
        "page_number": 9,
        "text": "Table 3: Variations on the Transformer architecture. Unlisted values are "
                "identical to those of the base model. All metrics are on the English-to-German "
                "translation development set, newstest2013. Listed perplexities are per-wordpiece, "
                "according to our"
    },
    {
        "page_number": 2,
        "text": "dge, however, the Transformer is the first transduction model relying entirely "
                "on self-attention to compute representations of its input and output without "
                "using sequence- aligned RNNs or convolution. In the following sections, we will "
                "describe the T"
    },
    {
        "page_number": 7,
        "text": "length n is smaller than the representation dimensionality d, which is most "
                "often the case with sentence representations used by state-of-the-art models "
                "in machine translations, such as word-piece [38] and byte-pair [31] "
                "representations. To improve c"
    },
]

COPYRIGHT_CHUNK = CHUNKS[0]

print(f"Query used for both conditions:\n  {QUESTION!r}\n")

print("=== Condition A: single chunk in isolation (copyright chunk only) ===")
for i in range(4):
    chunk_copy = dict(COPYRIGHT_CHUNK)  # fresh dict — check() mutates in place
    result = checker.check(QUESTION, [chunk_copy])
    print(f"Run {i + 1}: relevant = {result[0]['relevant']}")

print("\n=== Condition B: full 5-chunk batch (same query, same chunks as real run) ===")
for i in range(4):
    chunks_copy = [dict(c) for c in CHUNKS]  # fresh copies each run
    result = checker.check(QUESTION, chunks_copy)
    page1_result = next(c for c in result if c["page_number"] == 1)
    print(f"Run {i + 1}: Page 1 (copyright chunk) relevant = {page1_result['relevant']}")