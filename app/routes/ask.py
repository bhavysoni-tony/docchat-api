"""
Route for asking a question about an uploaded document: POST /ask.

Why this file exists:
This is the "retrieval + generation" half of the RAG flow — it takes a
question, finds relevant chunks in Qdrant, and asks Gemini to answer
using only those chunks.
"""

from fastapi import APIRouter, HTTPException

from app.models import AskRequest, AskResponse, Source
from app.metadata_store import get_document
from app.embeddings import embed_text
from app.vector_store import search_chunks
from app.llm import generate_answer

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    # --- Validate the question isn't blank/whitespace-only ---
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # --- Validate the document exists ---
    document = get_document(request.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    # --- Embed the question and search for relevant chunks ---
    query_vector = embed_text(request.question)
    chunks = search_chunks(request.document_id, query_vector, top_k=4)

    if not chunks:
        # No chunks at all for this document (shouldn't normally happen,
        # but guards against an edge case rather than crashing).
        return AskResponse(answer="Not found in this document.", sources=[])

    # --- Ask Gemini to answer using only the retrieved chunks ---
    answer = generate_answer(request.question, chunks)

    sources = [
        Source(page=chunk["page_number"], snippet=chunk["text"][:200])
        for chunk in chunks
    ]

    return AskResponse(answer=answer, sources=sources)
