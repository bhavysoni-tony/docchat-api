"""
Basic test for POST /ask.

Why this test exists:
It checks the endpoint's validation logic (empty question -> 400,
unknown document -> 404) without needing a real Gemini API key or a
populated Qdrant database. We mock get_document and the LLM/vector
calls so this test is fast and runs in CI with no external services.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask", json={"document_id": "some-id", "question": "   "}
    )
    assert response.status_code == 400


def test_ask_returns_404_for_unknown_document():
    with patch("app.routes.ask.get_document", return_value=None):
        response = client.post(
            "/ask", json={"document_id": "does-not-exist", "question": "What is this?"}
        )
    assert response.status_code == 404


def test_ask_returns_grounded_answer_for_known_document():
    fake_document = {"document_id": "abc123", "file_name": "test.pdf"}
    fake_chunks = [{"page_number": 1, "text": "Refunds are allowed within 30 days.", "score": 0.9}]

    with patch("app.routes.ask.get_document", return_value=fake_document), \
         patch("app.routes.ask.search_chunks", return_value=fake_chunks), \
         patch("app.routes.ask.generate_answer", return_value="Refunds are allowed within 30 days."):
        response = client.post(
            "/ask", json={"document_id": "abc123", "question": "What is the refund policy?"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Refunds are allowed within 30 days."
    assert body["sources"][0]["page"] == 1
