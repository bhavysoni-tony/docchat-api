"""
Routes for uploading and listing documents: POST /documents, GET /documents,
and DELETE /documents/{id} (bonus).

Why this file exists:
This is where the full "PDF -> text -> chunks -> embeddings -> Qdrant"
pipeline is wired together in order. Each step itself lives in its own
module (pdf_extraction, chunking, embeddings, vector_store) — this file
just calls them in sequence and turns errors into the right HTTP codes.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models import DocumentResponse
from app.pdf_extraction import extract_pages, NoReadableTextError
from app.chunking import chunk_pages
from app.embeddings import embed_texts
from app.vector_store import upsert_chunks, delete_document_chunks
from app.metadata_store import save_document, list_documents, get_document, delete_document

router = APIRouter()


@router.post("/documents", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    # --- Step 1: reject anything that isn't a PDF ---
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    # --- Step 2: extract text per page ---
    try:
        pages = extract_pages(pdf_bytes)
    except NoReadableTextError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # --- Step 3: split each page into overlapping chunks ---
    chunks = chunk_pages(pages)

    # --- Step 4: embed every chunk in one batch call ---
    chunk_texts = [c["text"] for c in chunks]
    vectors = embed_texts(chunk_texts)

    # --- Step 5: store vectors + metadata in Qdrant ---
    document_id = str(uuid.uuid4())
    upsert_chunks(document_id, chunks, vectors)

    # --- Step 6: store document-level metadata to disk (survives restart) ---
    document = {
        "document_id": document_id,
        "file_name": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "upload_time": datetime.now(timezone.utc).isoformat(),
    }
    save_document(document)

    return document


@router.get("/documents", response_model=list[DocumentResponse])
async def get_all_documents():
    return list_documents()


@router.delete("/documents/{document_id}")
async def remove_document(document_id: str):
    document = get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    delete_document_chunks(document_id)
    delete_document(document_id)
    return {"message": f"Document {document_id} deleted."}
