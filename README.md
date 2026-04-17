# RAG Backend — AI Tutor Search Service

PDF ingestion and semantic retrieval backend for a Socratic AI tutor.

**Live API:** `https://rag-aitutor-production.up.railway.app`  
**API Docs:** `https://rag-aitutor-production.up.railway.app/docs`  
**Agent Integration Guide:** [`AGENT_INTEGRATION.md`](./AGENT_INTEGRATION.md)

---

## What It Does

Accepts PDF course materials, extracts and indexes their text, and returns the most relevant chunks for any query. Built specifically to support a Socratic tutoring agent that retrieves course content before asking students questions.

**In scope:** PDF upload → text extraction → chunking → embedding → vector storage → semantic retrieval → automatic background OCR  
**Out of scope:** answer generation, Socratic logic, conversation memory, any user-facing UI

---

## Architecture

```
Upload
  POST /files/upload
    → validate (PDF, size, course exists)
    → save to persistent disk (/data/storage)
    → background ingestion:
        PyMuPDF → native text extraction
        pages with < 50 chars → recorded in empty_pages
        token-based chunking (400 tok / 50 overlap, cl100k_base)
        OpenAI text-embedding-3-small → 1536d vectors
        Qdrant + Postgres
    → status: indexed in ~2–3 seconds
    → auto-triggers background OCR immediately after indexed:
        GPT-4o-mini Vision on all empty_pages (up to 8 parallel)
        re-chunks, re-embeds, updates index
        ocr_completed=true when done (~30–90s depending on page count)

Retrieve
  POST /retrieve
    → embed query (OpenAI)
    → Qdrant cosine search, filtered by course_id
    → returns chunks with score, low_confidence flag, page number

On-Demand OCR (ad-hoc, single page)
  POST /files/{id}/ocr/page   → single page, synchronous (~4 sec)
    → GPT-4o-mini Vision
    → re-chunks, re-embeds, updates index
```

---

## Stack

| Component | Technology |
|---|---|
| API | FastAPI |
| Vector DB | Qdrant (named vector "dense", cosine) |
| Relational DB | PostgreSQL |
| Embeddings | OpenAI `text-embedding-3-small` (1536d) |
| OCR | OpenAI `gpt-4o-mini` Vision |
| PDF parsing | PyMuPDF (fitz) |
| Tokenizer | tiktoken `cl100k_base` |
| Hosting | Railway ($5/month) |
| Storage | Railway persistent Volume at `/data` |

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/courses` | Create a course |
| `GET` | `/courses/{id}/files` | List files in a course |
| `POST` | `/files/upload` | Upload and index a PDF |
| `GET` | `/files/{id}` | Poll indexing status |
| `DELETE` | `/files/{id}` | Delete file, chunks, and vectors |
| `POST` | `/files/{id}/reindex` | Re-run ingestion on existing file |
| `POST` | `/files/{id}/ocr/page` | OCR a single page on demand |
| `POST` | `/retrieve` | Semantic search across course files |

---

## File Status Flow

```
uploaded → processing → extracting → chunking → embedding → indexed
                                                           ↘ failed
```

| Status | Meaning |
|---|---|
| `uploaded` | Saved to disk, queued for ingestion |
| `processing` | Background task started |
| `extracting` | PyMuPDF reading pages |
| `chunking` | Text being split into chunks |
| `embedding` | OpenAI embedding in progress |
| `indexed` | Ready for retrieval |
| `failed` | Pipeline error — see `error_message` |

---

## File Response Fields

```json
{
  "file_id": "...",
  "status": "indexed",
  "chunk_count": 60,
  "total_page_count": 82,
  "empty_pages": [4, 9, 14, 78],
  "ocr_completed": false,
  "file_exists": true
}
```

- `empty_pages` — pages with < 50 chars of native text, not yet OCR'd
- `ocr_completed` — true when all empty pages have been processed
- `file_exists` — false means PDF was lost (should not happen with volume mounted)

---

## Retrieve Response

```json
{
  "results": [
    {
      "text": "...",
      "score": 0.61,
      "low_confidence": false,
      "page": 34,
      "file_id": "...",
      "filename": "calculus.pdf"
    }
  ],
  "has_unindexed_pages": true
}
```

- `score` — cosine similarity (0–1). Above 0.40 is usable.
- `low_confidence` — true when score is below threshold

---

## Quick Start (local)

```bash
cp .env.example .env
# Set DATABASE_URL, OPENAI_API_KEY, QDRANT_HOST

docker compose up -d   # starts Postgres + Qdrant

pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs at `http://localhost:8000/docs`

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | required | Postgres connection string |
| `OPENAI_API_KEY` | required | OpenAI API key |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `course_chunks` | Collection name |
| `STORAGE_PATH` | `/data/storage` | PDF storage (use Railway Volume at `/data`) |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |
| `MAX_PAGES` | `300` | Max pages per PDF |
| `TOP_K_DEFAULT` | `5` | Default retrieval results |
| `TOP_K_MAX` | `20` | Max top_k |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.35` | Below this → `low_confidence: true` |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests use SQLite in-memory. No Postgres or Qdrant required.

---

## Known Limitations

- **No authentication** — all endpoints are open. Add API key middleware before exposing publicly.
- **Single-process executor** — ingestion runs in a bounded thread pool (not Celery). Fine for low concurrency; migrate for high load.
- **Garbled native text** — pages that extract text but with broken Unicode or scan artifacts pass the OCR threshold and index with low-quality chunks. Retrieval score will be low and flagged.
