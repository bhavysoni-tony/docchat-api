"""
Tests for app/chunking.py.

Why these tests exist:
Chunking is the one piece of the RAG pipeline we wrote entirely by
hand (no library), so it's the piece most likely to have an off-by-one
bug — these tests check the boundary behavior directly.
"""

from app.chunking import chunk_text, chunk_pages


def test_chunk_text_empty_input_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short_text_returns_single_chunk():
    text = "This is a short sentence."
    result = chunk_text(text, chunk_size=800, overlap=100)
    assert result == [text]


def test_chunk_text_produces_no_empty_chunks():
    text = "word " * 500  # long enough to require multiple chunks
    result = chunk_text(text, chunk_size=800, overlap=100)
    assert all(chunk.strip() != "" for chunk in result)


def test_chunk_text_overlap_behavior():
    # Build text where each character position is identifiable.
    text = "".join(str(i % 10) for i in range(2000))
    chunk_size = 800
    overlap = 100

    result = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # The end of chunk 1 and the start of chunk 2 should share the
    # overlapping characters.
    end_of_first = text[chunk_size - overlap : chunk_size]
    start_of_second = result[1][: len(end_of_first)]
    assert end_of_first == start_of_second


def test_chunk_text_rejects_overlap_larger_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=150)


def test_chunk_pages_keeps_page_numbers():
    pages = [
        {"page_number": 1, "text": "word " * 300},
        {"page_number": 2, "text": "different word " * 300},
    ]
    result = chunk_pages(pages)

    page_numbers_seen = {c["page_number"] for c in result}
    assert page_numbers_seen == {1, 2}

    # Every chunk from page 1 should only contain page-1 text, i.e.
    # chunks never span across pages.
    for chunk in result:
        if chunk["page_number"] == 1:
            assert "different" not in chunk["text"]
