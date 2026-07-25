import os
import fitz

BOILERPLATE_MARKERS = [
    "hereby grants permission",
    "all rights reserved",
    "reproduced with permission",
    "no part of this publication may be reproduced",
    "for use in journalistic or scholarly works",
    "unauthorized reproduction",
]

def strip_boilerplate(text):
    """
    Remove lines matching known copyright/licensing boilerplate patterns
    from extracted PDF page text. Line-based, case-insensitive substring
    matching against a fixed set of common academic-publishing phrases.

    Known limitation: only strips lines that themselves contain a marker
    phrase. A boilerplate notice wrapped across multiple PDF-rendered
    lines may leave an orphaned fragment if only one of its lines matches.
    """
    lines = text.split("\n")
    kept_lines = [
        line for line in lines
        if not any(marker in line.lower() for marker in BOILERPLATE_MARKERS)
    ]
    return "\n".join(kept_lines)

def load_pdf(pdf_path):
    """
    Extract text from a PDF page by page.
    Store page number and source filename for citation.
    """

    document = fitz.open(pdf_path)
    source_name = os.path.basename(pdf_path)

    pages = []

    for page_num in range(len(document)):
        page = document[page_num]
        text = page.get_text().strip()
        text = strip_boilerplate(text)

        if text:
            pages.append(
                {
                    "source": source_name,
                    "page_number": page_num + 1,
                    "text": text
                }
            )

    return pages