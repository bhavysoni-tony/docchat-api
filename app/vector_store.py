"""
Handles all interaction with Qdrant (the vector database).

Why this file exists:
This is the only file that imports qdrant_client. Routes and other
modules call simple functions like upsert_chunks() and search_chunks()
without needing to know Qdrant's API — if we ever swapped Qdrant for
another vector database, only this file would change.
"""

import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import QDRANT_URL, QDRANT_COLLECTION_NAME
from app.embeddings import get_embedding_dimension

# QdrantClient(path=...) runs Qdrant in "local mode": it stores its data
# as files on disk at this path, with no separate server process needed.
# This satisfies "persistent storage" with the least possible setup —
# the alternative (QdrantClient(url="http://localhost:6333")) talks to
# a Qdrant server, e.g. running in Docker (see docker-compose.yml).
_client = QdrantClient(path=QDRANT_URL)


def _ensure_collection_exists() -> None:
    """Creates the Qdrant collection on first use if it doesn't already exist."""
    existing = [c.name for c in _client.get_collections().collections]
    if QDRANT_COLLECTION_NAME not in existing:
        _client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=get_embedding_dimension(),
                distance=qmodels.Distance.COSINE,
            ),
        )


def upsert_chunks(document_id: str, chunks: list[dict], vectors: list[list[float]]) -> None:
    """
    Stores chunk vectors in Qdrant, tagged with document_id and page_number.

    Args:
        document_id: the parent document's ID, so we can later filter
            search results down to "only chunks from this document".
        chunks: list of {"page_number": int, "text": str}.
        vectors: embedding for each chunk, same order as `chunks`.
    """
    _ensure_collection_exists()

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                },
            )
        )

    _client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)


def search_chunks(document_id: str, query_vector: list[float], top_k: int = 4) -> list[dict]:
    """
    Finds the most similar chunks to a query vector, restricted to one document.

    Args:
        document_id: only chunks belonging to this document are searched.
            This filter matters — without it, a question about Document A
            could retrieve context from Document B.
        query_vector: the embedded question.
        top_k: how many chunks to retrieve.

    Returns:
        A list of {"page_number": int, "text": str, "score": float}.
    """
    results = _client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(value=document_id),
                )
            ]
        ),
        limit=top_k,
    )

    return [
        {
            "page_number": r.payload["page_number"],
            "text": r.payload["text"],
            "score": r.score,
        }
        for r in results
    ]


def delete_document_chunks(document_id: str) -> None:
    """Deletes all chunks belonging to a document (used by DELETE /documents/{id})."""
    _client.delete(
        collection_name=QDRANT_COLLECTION_NAME,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )
        ),
    )
