# RAG Backend Service

Document ingestion and retrieval service for an AI tutor system.

Handles: PDF upload → text extraction → chunking → embedding → vector storage → semantic retrieval.

Does **not** handle: answer generation, tutor logic, Socratic modes, quiz generation, Telegram, or any user-facing interface.

---

## Architecture

```
POST /files/upload
  → validate → save PDF → create DB record → background ingestion task
  → extract text (PyMuPDF) → chunk (sentence-aware) → embed → Qdrant + Postgres
  → status: uploaded → processing → indexed | failed

POST /retrieve
  → embed query → Qdrant search (filtered by course_id) → Postgres metadata lookup
  → return ranked chunks with page, filename, score, low_confidence flag
```

## Authorization model

This service does not enforce caller authorization. It assumes all requests come from a trusted upstream agent. It must not be publicly routable without a gateway or network-level access control.

## Consistency model

No cross-system transactions between Postgres and Qdrant. Ordering of operations (Qdrant first on insert, Qdrant first on delete) minimizes orphan risk. Startup recovery re-queues files stuck in `processing` after a crash.

---

## Quick start

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL and OPENAI_API_KEY

docker compose up -d

pip install -r requirements.txt

# Create tables (or run Alembic migrations)
python -c "from app.database import engine; from app.models import Base; Base.metadata.create_all(engine)"

uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

---

## Alembic migrations

```bash
# Generate initial migration from current models
alembic revision --autogenerate -m "initial"

# Apply migrations
alembic upgrade head
```

---

## Running tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

Tests use SQLite in-memory. No running Postgres or Qdrant required.

---

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | required | Postgres connection string |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `course_chunks` | Collection name |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `local` |
| `OPENAI_API_KEY` | required if openai | OpenAI API key |
| `STORAGE_PATH` | `./storage` | Local PDF storage directory |
| `MAX_FILE_SIZE_MB` | `50` | Upload size cap |
| `MAX_PAGES` | `300` | Max PDF pages (total, not extractable) |
| `TOP_K_DEFAULT` | `5` | Default retrieval results |
| `TOP_K_MAX` | `20` | Maximum top_k allowed |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.70` | Below this score → `low_confidence: true` |
| `MIN_EXTRACTABLE_RATIO` | `0.50` | Fail ingestion if fewer than 50% of pages have text |

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/courses` | Create a course |
| `GET` | `/courses/{course_id}/files` | List files in a course |
| `POST` | `/files/upload` | Upload a PDF for ingestion |
| `GET` | `/files/{file_id}` | Poll ingestion status |
| `DELETE` | `/files/{file_id}` | Delete file + chunks + vectors |
| `POST` | `/retrieve` | Retrieve relevant chunks for a query |
| `GET` | `/health` | Health check |

---

## File statuses

| Status | Meaning |
|---|---|
| `uploaded` | Saved to disk, ingestion not yet started |
| `processing` | Background ingestion task is running |
| `indexed` | All chunks embedded and stored |
| `failed` | Pipeline error — see `error_message` |

---

## Known limitations (MVP)

- **BackgroundTasks**: ingestion runs in the web server process. Migrate to Celery/ARQ for production.
- **Chunker**: regex sentence splitting is a baseline. Replace with `nltk` or `spacy` for robustness.
- **No retry**: failed files must be re-uploaded manually.
- **Local storage**: PDFs are stored on local disk. Replace `storage_path` logic with S3/GCS for multi-instance deployments.
- **No auth**: trusted upstream model only. Add API key middleware before exposing beyond private network.
