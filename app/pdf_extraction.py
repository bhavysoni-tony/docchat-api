"""
Extracts text from a PDF file, page by page.

Why this file exists:
Before we can chunk or embed anything, we need plain text pulled out of
the PDF. We extract page-by-page (rather than one giant blob) because
the assignment requires every answer to cite a page number — that's
only possible if we know which page each piece of text came from.
"""

import pymupdf


class NoReadableTextError(Exception):
    """Raised when a PDF has zero extractable text (e.g. scanned images only)."""

    pass


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """
    Extracts text from each page of a PDF.

    Args:
        pdf_bytes: the raw bytes of the uploaded PDF file.

    Returns:
        A list of dicts, one per page, e.g.:
        [{"page_number": 1, "text": "..."}, {"page_number": 2, "text": "..."}]
        Empty pages (no text) are skipped.

    Raises:
        NoReadableTextError: if none of the pages contain any text at all.
    """
    # stream=pdf_bytes lets us open the PDF directly from memory —
    # we never need to save the upload to disk first.
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text().strip()
        if text:
            # PDF pages are 0-indexed internally; humans count from page 1.
            pages.append({"page_number": page_index + 1, "text": text})

    doc.close()

    if not pages:
        raise NoReadableTextError(
            "No readable text found in this PDF. It may be a scanned "
            "image without a text layer."
        )

    return pages
