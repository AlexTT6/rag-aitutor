# RAG Backend — Agent Integration Guide

This document is for the Socratic agent developer.
The RAG backend is a deployed REST API. The agent makes HTTP calls to it — nothing else is needed.

---

## Base URL

```
https://rag-aitutor-production.up.railway.app
```

---

## Existing Course and File (for testing)

```
course_id: 319cc001-692c-4078-9bc7-84e67367ed9b
file_id:   3adb50cc-df18-4f12-ae71-1ade493ff38b  (82-page calculus PDF, 60 chunks)
```

---

## The Three Endpoints You Need

### 1. Retrieve — called on every student message

```
POST /retrieve
```

**Request:**
```json
{
  "query": "student message text here",
  "course_id": "319cc001-692c-4078-9bc7-84e67367ed9b",
  "top_k": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "actual text from the textbook...",
      "score": 0.61,
      "low_confidence": false,
      "page": 34,
      "file_id": "3adb50cc-...",
      "filename": "calculus.pdf"
    }
  ],
  "has_unindexed_pages": true
}
```

**What to check:**
- `results[0].score` — quality of the match (0.0 to 1.0)
- `low_confidence: true` — score is below threshold, result may be weak
- `has_unindexed_pages: true` — some pages were not OCR'd yet, better results possible after OCR

---

### 2. OCR Missing Pages — trigger in background when results are weak

```
POST /files/{file_id}/ocr/missing
```

No request body needed.

**Response (immediate, 202):**
```json
{
  "status": "started",
  "pages_count": 22,
  "message": "OCR started for 22 pages. Poll GET /files/{file_id} to monitor."
}
```

OCR runs in the background (~130 seconds for 22 pages).
**Do not wait for it.** Fire and continue the conversation.

---

### 3. OCR Single Page — when student asks about a specific page

```
POST /files/{file_id}/ocr/page
```

**Request:**
```json
{
  "page": 78
}
```

**Response (~4 seconds, synchronous):**
```json
{
  "page": 78,
  "indexed": true,
  "ocr_text_length": 429,
  "message": "Page 78 OCR'd and added to index."
}
```

After this returns, call `/retrieve` again — the page is now in the index.

---

## Integration Logic

```python
import requests

BASE_URL = "https://rag-aitutor-production.up.railway.app"
COURSE_ID = "319cc001-692c-4078-9bc7-84e67367ed9b"
FILE_ID   = "3adb50cc-df18-4f12-ae71-1ade493ff38b"

SCORE_THRESHOLD = 0.40  # below this → results are weak


def retrieve(query: str) -> dict:
    r = requests.post(f"{BASE_URL}/retrieve", json={
        "query": query,
        "course_id": COURSE_ID,
        "top_k": 5,
    }, timeout=10)
    r.raise_for_status()
    return r.json()


def ocr_missing():
    """Fire and forget — do not await."""
    requests.post(f"{BASE_URL}/files/{FILE_ID}/ocr/missing", timeout=10)


def ocr_page(page_num: int):
    """Synchronous — waits ~4 seconds."""
    r = requests.post(f"{BASE_URL}/files/{FILE_ID}/ocr/page",
                      json={"page": page_num}, timeout=30)
    r.raise_for_status()
    return r.json()


def get_context(student_message: str) -> tuple[str, bool]:
    """
    Returns (context_text, is_strong).
    Call this on every student message.
    """
    response = retrieve(student_message)
    results = response.get("results", [])
    has_unindexed = response.get("has_unindexed_pages", False)

    if not results:
        return "", False

    top_score = results[0]["score"]
    is_strong = top_score >= SCORE_THRESHOLD

    if not is_strong and has_unindexed:
        ocr_missing()  # start OCR in background, don't wait

    context = "\n\n".join(r["text"] for r in results)
    return context, is_strong
```

---

## Socratic Flow with RAG

```
Student: "I don't understand limits"
    │
    ▼
context, is_strong = get_context("I don't understand limits")
    │
    ├── is_strong = True  →  use context, ask Socratic question based on real text
    │
    └── is_strong = False →  ocr_missing() fired in background
                             ask opening Socratic question to student:
                             "What do you already know about this topic?"

Student: "I think it's when x goes to infinity..."
    │
    ▼
context, is_strong = get_context("x goes to infinity limits")
    │
    ├── is_strong = True  →  compare student's answer to context
    │                        correct → ask deeper question
    │                        incorrect → ask guiding question toward correct answer
    │
    └── is_strong = False →  OCR may still be running
                             use what's available, ask another Socratic question
```

**Rule:** Call `get_context()` on every student message except pure acknowledgements ("ok", "thanks", "got it").

---

## Checking OCR Status (optional)

If you want to know when OCR is done:

```
GET /files/{file_id}
```

```json
{
  "status": "indexed",
  "ocr_completed": true,
  "chunk_count": 76,
  "empty_pages": null
}
```

`ocr_completed: true` means all pages are indexed. At this point `/retrieve` returns the best possible results.

---

## File Upload (if you need to add new documents)

```
POST /files/upload
Content-Type: multipart/form-data

file:      <pdf file>
course_id: 319cc001-692c-4078-9bc7-84e67367ed9b
```

Then poll `GET /files/{file_id}` until `status: "indexed"` (usually 2–5 seconds for a normal PDF).

---

## Score Reference

| Score | Meaning | What to do |
|-------|---------|------------|
| ≥ 0.60 | Strong match | Use directly |
| 0.40 – 0.59 | Moderate match | Use, but verify with Socratic question |
| < 0.40 | Weak match | Trigger OCR if `has_unindexed_pages`, ask opening question |
| 0 results | Nothing found | Ask student to rephrase |

---

## What the Agent Does NOT Need to Know

- How Qdrant or Postgres work
- How embeddings are generated
- How OCR processes pages
- Anything about chunking or tokenization

The RAG backend handles all of that. The agent only needs to call `/retrieve`, check the score, and decide what question to ask next.
