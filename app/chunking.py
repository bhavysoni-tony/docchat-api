"""
Splits page text into overlapping chunks, manually (no LangChain).

Why this file exists:
Embedding models and LLMs work best on small, focused pieces of text —
not entire pages. We split each page's text into fixed-size chunks with
a small overlap so a fact that happens to sit right at a chunk boundary
still appears in full in at least one chunk.

Chunk size and overlap chosen (explained fully in README):
- CHUNK_SIZE = 800 characters: roughly 150-200 words, small enough to be
  a precise, focused unit for embedding search, large enough to hold a
  complete idea or a couple of sentences of context.
- CHUNK_OVERLAP = 100 characters: about 12% overlap. Enough that a
  sentence split across two chunks still appears complete in one of
  them, without duplicating so much text that storage/search cost
  balloons.
"""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits a string into overlapping chunks of roughly `chunk_size` characters.

    Args:
        text: the text to split (already extracted from a PDF page).
        chunk_size: target number of characters per chunk.
        overlap: number of characters repeated at the start of the next
            chunk, taken from the end of the previous one.

    Returns:
        A list of non-empty chunk strings. Returns an empty list if
        `text` is empty or only whitespace.
    """
    text = text.strip()
    if not text:
        return []

    if overlap >= chunk_size:
        # Guard against a nonsensical config that would loop forever
        # (the window would never move forward).
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move the window forward by (chunk_size - overlap), not by
        # chunk_size, so the next chunk re-includes the last `overlap`
        # characters of this one.
        start += chunk_size - overlap

    return chunks


def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Chunks every page independently, keeping each chunk tied to its page number.

    We chunk per-page rather than joining all pages into one big string
    first — that would let a single chunk silently span two different
    pages, and we would lose the ability to report an accurate page
    number for that chunk's source citation.

    Args:
        pages: output of pdf_extraction.extract_pages(), e.g.
            [{"page_number": 1, "text": "..."}, ...]

    Returns:
        A list of dicts: [{"page_number": 1, "text": "chunk text"}, ...]
    """
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(page["text"], chunk_size, overlap)
        for chunk in page_chunks:
            all_chunks.append({"page_number": page["page_number"], "text": chunk})

    return all_chunks
