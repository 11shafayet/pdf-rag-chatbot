import os
import fitz


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

        if text:
            pages.append(
                {
                    "source": source_name,
                    "page_number": page_num + 1,
                    "text": text
                }
            )

    return pages