import fitz


def load_pdf(pdf_path):
    """
    Extract text from a PDF page by page.
    """

    document = fitz.open(pdf_path)

    pages = []

    for page_num in range(len(document)):
        page = document[page_num]

        pages.append(
            {
                "page_number": page_num + 1,
                "text": page.get_text()
            }
        )

    return pages