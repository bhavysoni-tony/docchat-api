"""
Converts text into embedding vectors using Sentence Transformers.

Why this file exists:
Qdrant stores and searches vectors, not raw text. This is the one place
in the codebase that turns text into numbers, so both the upload path
(embedding chunks) and the ask path (embedding the question) call the
same function and stay consistent — if they used different models, the
vectors would live in different "spaces" and similarity search would
return garbage.
"""

from sentence_transformers import SentenceTransformer

# "all-MiniLM-L6-v2" is a small, fast, well-known general-purpose model
# (384-dimensional vectors). It's a common practical default for RAG
# projects because it runs on CPU quickly with good enough quality —
# not because it's the "best" embedding model available.
MODEL_NAME = "all-MiniLM-L6-v2"

# Loaded once at import time and reused for every request. Loading a
# model from disk takes a second or two — we do not want to pay that
# cost on every single API call.
_model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """Embeds a single string into a vector."""
    vector = _model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of strings at once (more efficient than one-by-one)."""
    vectors = _model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def get_embedding_dimension() -> int:
    """Returns the vector size this model produces (needed to create the Qdrant collection)."""
    return _model.get_sentence_embedding_dimension()
