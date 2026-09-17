"""
FastAPI application entrypoint.

Why this file exists:
This is what `uvicorn app.main:app` actually runs. It creates the
FastAPI app, validates configuration at startup, and includes the
route modules. It intentionally contains no business logic itself —
that all lives in routes/ and the pipeline modules.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import validate_config
from app.routes import documents, ask

# Fail fast: if GEMINI_API_KEY is missing, the server won't start at
# all, instead of starting successfully and failing later on the first
# real request.
validate_config()

app = FastAPI(
    title="DocChat API",
    description=(
        "Upload a PDF, then ask questions about it. Answers are grounded "
        "only in the document's content using a retrieval-augmented "
        "generation (RAG) pipeline: PyMuPDF for text extraction, manual "
        "chunking, Sentence Transformers for embeddings, Qdrant for "
        "vector search, and Gemini for answer generation."
    ),
    version="1.0.0",
)

app.include_router(documents.router, tags=["Documents"])
app.include_router(ask.router, tags=["Ask"])


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Catches any error we didn't explicitly handle (e.g. a bug, a
    third-party API outage) and returns a generic 500 message instead
    of leaking a stack trace or internal details to the client.
    Deliberate HTTPExceptions (400, 404, etc.) raised in routes are
    NOT caught here — FastAPI handles those separately and correctly.
    """
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


@app.get("/", tags=["Health"])
async def root():
    """Simple health check / landing route."""
    return {"status": "ok", "message": "DocChat API is running. See /docs for the API reference."}
