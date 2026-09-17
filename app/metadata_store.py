"""
Stores document metadata (id, filename, page count, etc.) in a JSON file.

Why this file exists:
Qdrant stores chunk vectors, but GET /documents needs to list documents
even before any question is asked — that's document-level metadata, not
chunk-level, so it doesn't naturally belong in Qdrant. A single JSON
file is the simplest thing that satisfies "must survive a server
restart" without introducing a real database (which the assignment
explicitly says to avoid unless necessary).
"""

import json
import os
from threading import Lock

METADATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.json")

# Guards against two requests writing to the file at the exact same
# moment and corrupting it. FastAPI can handle requests concurrently,
# so without this, two simultaneous uploads could interleave their
# writes to the same file.
_lock = Lock()


def _read_all() -> dict:
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def _write_all(data: dict) -> None:
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_document(document: dict) -> None:
    """Adds or updates one document's metadata. `document` must include 'document_id'."""
    with _lock:
        data = _read_all()
        data[document["document_id"]] = document
        _write_all(data)


def get_document(document_id: str) -> dict | None:
    """Returns one document's metadata, or None if it doesn't exist."""
    data = _read_all()
    return data.get(document_id)


def list_documents() -> list[dict]:
    """Returns all documents' metadata, most recently uploaded first."""
    data = _read_all()
    documents = list(data.values())
    documents.sort(key=lambda d: d["upload_time"], reverse=True)
    return documents


def delete_document(document_id: str) -> None:
    """Removes one document's metadata entry."""
    with _lock:
        data = _read_all()
        data.pop(document_id, None)
        _write_all(data)
