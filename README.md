# DocChat API

A backend API that lets you upload a PDF and ask questions about it, answered
using only that document's own content (Retrieval-Augmented Generation).

## 1. Project Overview

You upload a PDF. The server extracts its text, splits it into small
overlapping chunks, embeds those chunks into vectors, and stores them in a
vector database (Qdrant). When you ask a question, the server embeds your
question, finds the most similar chunks in Qdrant, and sends only those
chunks (plus your question) to Gemini with an instruction to answer strictly
from that context — not from the model's general knowledge.

## 2. Architecture

```
Upload flow:
  PDF file -> extract text per page (PyMuPDF)
           -> split into overlapping chunks (manual chunking)
           -> embed chunks (Sentence Transformers)
           -> store vectors + page number in Qdrant
           -> store document metadata in metadata.json

Question flow:
  Question -> embed question (Sentence Transformers)
           -> similarity search in Qdrant, filtered to one document
           -> top matching chunks
           -> prompt Gemini: "answer only from this context"
           -> grounded answer + source pages/snippets
```

## 3. Features

- Upload a PDF and extract its text, page by page
- Manual text chunking (no LangChain text splitters)
- Semantic search over chunks using vector embeddings
- Grounded question answering — Gemini is told to only use retrieved context
- Answers include source page numbers and snippets
- Document metadata persists across server restarts
- Clear HTTP error codes for invalid input
- Basic automated tests (chunking + `/ask` validation)
- Docker/Docker Compose setup
- `DELETE /documents/{id}` bonus endpoint

## 4. Tech Stack

| Purpose              | Tool                          |
|----------------------|--------------------------------|
| Web framework        | FastAPI + Uvicorn              |
| PDF text extraction  | PyMuPDF                        |
| Embeddings           | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector database      | Qdrant (local persistent mode) |
| LLM                  | Gemini (`gemini-1.5-flash`)    |
| Validation           | Pydantic                       |
| Config               | python-dotenv                  |
| Testing              | pytest                         |

## 5. Project Structure

```
docchat-api/
├── app/
│   ├── main.py              # FastAPI app creation, router wiring, global error handler
│   ├── config.py            # loads and validates environment variables
│   ├── models.py            # Pydantic request/response schemas
│   ├── pdf_extraction.py    # PDF -> per-page text (PyMuPDF)
│   ├── chunking.py          # manual overlapping chunk splitter
│   ├── embeddings.py        # text -> vector (Sentence Transformers)
│   ├── vector_store.py      # all Qdrant reads/writes
│   ├── metadata_store.py    # JSON-file document metadata persistence
│   ├── llm.py                # builds the grounded prompt, calls Gemini
│   └── routes/
│       ├── documents.py     # POST/GET /documents, DELETE /documents/{id}
│       └── ask.py            # POST /ask
├── tests/
│   ├── test_chunking.py     # chunking correctness tests
│   └── test_ask.py           # /ask endpoint validation tests (mocked)
├── data/                     # qdrant_storage/ and metadata.json live here (gitignored)
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

**Why this structure:** each file is one stage of the RAG pipeline. There
are no service/repository/factory layers — for a project this size they'd
add indirection without adding clarity, and the assignment specifically
asks for something a junior/mid-level engineer would write and be able to
explain line by line.

## 6. Installation

```bash
git clone <your-repo-url>
cd docchat-api
py -3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 7. Environment Variables

Copy the example file and fill in your Gemini key:

```bash
cp .env.example .env
```

```
GEMINI_API_KEY=your_key_here
QDRANT_URL=./data/qdrant_storage
QDRANT_COLLECTION_NAME=docchat_chunks
```

`QDRANT_URL` here is a **local folder path**, not a network URL — see the
next section.

## 8. How to Run Qdrant

Two options, both persistent:

**Option A — Local mode (default, no Docker needed):**
`QdrantClient(path=QDRANT_URL)` in `vector_store.py` stores Qdrant's data as
files under `./data/qdrant_storage`. Nothing to install or run separately —
this is the simplest option and what `.env.example` is set up for.

**Option B — Docker container:**
```bash
docker run -p 6333:6333 -v $(pwd)/data/qdrant_storage:/qdrant/storage qdrant/qdrant
```
If you use this option, change `QDRANT_URL` in `.env` to `http://localhost:6333`
and update `vector_store.py` to use `QdrantClient(url=QDRANT_URL)` instead of
`QdrantClient(path=QDRANT_URL)`.

## 9. How to Start FastAPI

```bash
uvicorn app.main:app --reload
```

The server starts at `http://localhost:8000`.

## 10. Swagger Documentation URL

Interactive API docs (auto-generated by FastAPI): **http://localhost:8000/docs**

## 11. API Endpoint Examples

**Upload a document**
```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@example.pdf"
```
```json
{
  "document_id": "b3f1...",
  "file_name": "example.pdf",
  "pages": 5,
  "chunks": 12,
  "upload_time": "2026-09-17T10:00:00+00:00"
}
```

**List documents**
```bash
curl http://localhost:8000/documents
```

**Ask a question**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"document_id": "b3f1...", "question": "What is the refund policy?"}'
```
```json
{
  "answer": "The refund window is 30 days from delivery.",
  "sources": [
    {"page": 4, "snippet": "Refunds can be requested within 30 days..."}
  ]
}
```

**Delete a document (bonus)**
```bash
curl -X DELETE http://localhost:8000/documents/b3f1...
```

## 12. Chunk Size and Overlap

- **Chunk size: 800 characters** (~150-200 words)
- **Overlap: 100 characters** (~12.5%)

## 13. Why These Values

800 characters is large enough to usually contain a complete thought or a
couple of full sentences (which gives the embedding model enough context to
produce a meaningful vector), but small enough to keep each chunk focused on
one topic — a common failure mode in RAG is chunks so large that they blend
multiple topics together, which hurts retrieval precision.

100 characters of overlap protects against the case where an important
sentence happens to fall right across a chunk boundary and would otherwise
be split in half in *every* chunk. It's kept small (12.5%, not 50%) because
overlap directly increases how much duplicate text gets stored and searched
— more overlap isn't free.

These are reasonable defaults for general prose (contracts, policies,
manuals). A codebase or a table-heavy PDF might call for different values;
that trade-off is discussed with whoever owns the product decision, not
baked in as a universal constant.

## 14. RAG Flow Explanation

1. **Extraction** (`pdf_extraction.py`) — PyMuPDF reads the PDF and returns
   text per page, so every later chunk can be traced back to a page number.
2. **Chunking** (`chunking.py`) — each page's text is split into overlapping
   800-character windows using a hand-written sliding-window function.
3. **Embedding** (`embeddings.py`) — each chunk is converted into a 384-dim
   vector using `all-MiniLM-L6-v2`.
4. **Storage** (`vector_store.py`) — vectors are upserted into Qdrant with
   `document_id` and `page_number` as payload, so search can later be
   filtered to one document.
5. **Retrieval** — the question is embedded with the same model, and Qdrant
   returns the top-4 most similar chunks *for that document only*.
6. **Generation** (`llm.py`) — the retrieved chunks are inserted into a
   prompt template that instructs Gemini to answer only from that context,
   and to say "Not found in this document." if it can't.

## 15. Error Handling

| Condition                          | Response                          |
|-------------------------------------|------------------------------------|
| Uploaded file isn't a PDF           | `400` with a clear message        |
| PDF has no extractable text         | `400` with a clear message        |
| Empty/whitespace-only question      | `400` with a clear message        |
| Unknown `document_id`               | `404`                              |
| Missing `GEMINI_API_KEY`            | Server refuses to start, with a clear error (not a silent failure) |
| Any other unexpected error          | Generic `500`, no stack trace or internals leaked to the client |

Validation errors are raised as `HTTPException` inside the route functions
and are never caught by the global exception handler — only genuinely
unexpected exceptions fall through to the generic 500 handler in `main.py`.

## 16. How Data Survives Restart

- **Chunk vectors**: Qdrant's local mode writes to disk at
  `./data/qdrant_storage` on every upsert — nothing is kept only in memory.
- **Document metadata**: written to `./data/metadata.json` on every upload,
  read back fresh on every `GET /documents` call. No in-memory cache that
  could go stale or get lost on restart.

## 17. What I'd Change for 100 Concurrent Large-PDF Uploads

The current design processes a PDF fully (extract → chunk → embed → store)
inside the HTTP request itself, which is fine for a take-home but won't
hold up under real concurrent load. At scale I would:

- **Move processing off the request path**: push uploads onto a job queue
  (e.g. Celery + Redis, or SQS) and have background workers do extraction,
  chunking, and embedding. `POST /documents` would return immediately with
  a "processing" status; the client polls or gets a webhook when it's ready.
- **Object storage for the raw PDFs**: store uploaded files in S3 (or
  equivalent) instead of holding them only in request memory, so workers can
  pick them up independently of the web server.
- **Horizontally scale FastAPI workers**: run multiple Uvicorn/Gunicorn
  workers behind a load balancer, since the API layer itself should stay
  stateless and cheap once heavy work is offloaded to workers.
- **Managed/clustered Qdrant**: move from local-file mode to Qdrant Cloud or
  a clustered self-hosted deployment, since 100 concurrent writers to a
  single local file store would serialize and become a bottleneck.
- **Replace the JSON metadata file** with a real database (e.g. Postgres) —
  a single JSON file with a lock is fine for a demo, but not safe for many
  concurrent writers.
- **Add monitoring and rate limiting**: track embedding/LLM latency and
  error rates, and rate-limit uploads per user so one client can't exhaust
  Gemini API quota or Qdrant capacity for everyone else.

## 18. Known Limitations / Unfinished Items

- No authentication/authorization — anyone with API access can upload or
  query any document. Not implemented because the assignment scope is the
  RAG pipeline itself, not auth.
- No retry/backoff around the Gemini API call — a transient network error
  will currently surface as a generic 500 rather than being retried.
- Chunking is character-based, not token-based, so chunk sizes are only an
  approximation of how large the actual model context usage will be.
- Only PDFs with a text layer are supported — scanned image-only PDFs are
  correctly rejected but not OCR'd (OCR was out of scope).
- No pagination on `GET /documents` — fine at small scale, would need
  pagination if the metadata store grew large.

## Running Tests

```bash
pytest
```

## Running with Docker

```bash
docker compose up --build
```
