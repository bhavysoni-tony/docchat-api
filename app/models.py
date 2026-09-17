"""
Pydantic models for API request and response bodies.

Why this file exists:
FastAPI uses these classes to automatically validate incoming JSON,
reject malformed requests with a 422 before our code even runs, and
generate the Swagger (/docs) schema. Keeping them in one file makes it
easy to see every shape the API accepts or returns.
"""

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Returned after a successful PDF upload, and as items in the list endpoint."""

    document_id: str
    file_name: str
    pages: int
    chunks: int
    upload_time: str


class AskRequest(BaseModel):
    """Body for POST /ask."""

    document_id: str
    question: str = Field(..., min_length=1)


class Source(BaseModel):
    """A single retrieved chunk used as evidence for an answer."""

    page: int
    snippet: str


class AskResponse(BaseModel):
    """Returned by POST /ask."""

    answer: str
    sources: list[Source]
