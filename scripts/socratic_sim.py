"""
Socratic Agent Simulator
========================
A deterministic test script that validates RAG backend behavior by simulating
a minimal Socratic-style tutor interaction.

NOT a real agent. No LLM. No memory. No conversation loop.
Purpose: verify retrieval quality, OCR fallback, and score improvement.

Usage:
    python scripts/socratic_sim.py \
        --url  https://rag-aitutor-production.up.railway.app \
        --course <course_id> \
        --file  <file_id_or_path_to_pdf> \
        --question "Explain the theorem on page 78"

    # Or use all defaults from the CONFIG block below.
"""

import argparse
import re
import sys
import time

import requests

# ---------------------------------------------------------------------------
# CONFIG — edit these or pass via CLI args
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL  = "https://rag-aitutor-production.up.railway.app"
DEFAULT_COURSE_ID = ""          # UUID — required
DEFAULT_FILE_ID   = ""          # UUID — if set, skip upload
DEFAULT_PDF_PATH  = ""          # local path — used only when FILE_ID is empty
DEFAULT_QUESTION  = "Explain the theorem on page 78"

STRONG_THRESHOLD  = 0.6         # score above this → strong result
WEAK_THRESHOLD    = 0.4         # all scores below this → weak result
TOP_K             = 5
POLL_INTERVAL_SEC = 3           # seconds between status polls
POLL_TIMEOUT_SEC  = 300         # give up after this many seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def _step(n: int, text: str) -> None:
    print(f"\n[Step {n}] {text}")


def _ok(text: str) -> None:
    print(f"  ✅  {text}")


def _warn(text: str) -> None:
    print(f"  ⚠️   {text}")


def _info(text: str) -> None:
    print(f"       {text}")


def _extract_page_number(question: str) -> int | None:
    """Extract a page number from the question string.
    Matches patterns like: 'page 78', 'страница 78', 'p.78', '#78'.
    """
    patterns = [
        r"page\s+(\d+)",
        r"страниц[еуа]?\s+(\d+)",
        r"p\.?\s*(\d+)",
        r"#\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _scores_summary(results: list) -> str:
    if not results:
        return "no results"
    scores = [r["score"] for r in results]
    return (
        f"top={scores[0]:.3f}  "
        f"min={min(scores):.3f}  "
        f"max={max(scores):.3f}  "
        f"count={len(scores)}"
    )


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def upload_pdf(base_url: str, course_id: str, pdf_path: str) -> str:
    """Upload a PDF and return the file_id."""
    _info(f"Uploading {pdf_path} ...")
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{base_url}/files/upload",
            data={"course_id": course_id},
            files={"file": (pdf_path.split("/")[-1], f, "application/pdf")},
            timeout=60,
        )
    resp.raise_for_status()
    file_id = resp.json()["file_id"]
    _ok(f"Uploaded → file_id={file_id}")
    return file_id


def poll_until_indexed(base_url: str, file_id: str) -> dict:
    """Poll GET /files/{id} until status is 'indexed' or 'failed'. Returns final file dict."""
    _info("Polling for indexed status ...")
    deadline = time.monotonic() + POLL_TIMEOUT_SEC
    last_status = None
    while time.monotonic() < deadline:
        resp = requests.get(f"{base_url}/files/{file_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status != last_status:
            _info(f"  status={status}")
            last_status = status
        if status == "indexed":
            return data
        if status == "failed":
            raise RuntimeError(f"Ingestion failed: {data.get('error_message')}")
        time.sleep(POLL_INTERVAL_SEC)
    raise TimeoutError(f"File not indexed after {POLL_TIMEOUT_SEC}s")


def retrieve(base_url: str, course_id: str, question: str) -> dict:
    """Call /retrieve and return full response dict."""
    resp = requests.post(
        f"{base_url}/retrieve",
        json={"query": question, "course_id": course_id, "top_k": TOP_K},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def trigger_page_ocr(base_url: str, file_id: str, page_num: int) -> dict:
    """POST /files/{id}/ocr/page — synchronous, waits until OCR is done."""
    resp = requests.post(
        f"{base_url}/files/{file_id}/ocr/page",
        json={"page": page_num},
        timeout=60,  # single OCR page takes ~4 sec
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def classify_results(results: list) -> str:
    """Returns 'strong', 'weak', or 'empty'."""
    if not results:
        return "empty"
    top_score = results[0]["score"]
    all_weak = all(r["score"] < WEAK_THRESHOLD for r in results)
    if top_score >= STRONG_THRESHOLD:
        return "strong"
    if all_weak:
        return "weak"
    return "moderate"


def make_socratic_question(chunk: dict) -> str:
    """Generate a deterministic Socratic question from a chunk."""
    text_snippet = chunk["text"][:120].strip().replace("\n", " ")
    page = chunk.get("page", "?")
    templates = [
        f'Based on page {page}: "{text_snippet}..." — what does this imply about the concept being described?',
        f'Your textbook says on page {page}: "{text_snippet}..." — can you restate this in your own words?',
        f'Look at page {page}. Given "{text_snippet}..." — what would happen if this condition were removed?',
    ]
    # Deterministic: pick by score
    idx = int(chunk["score"] * 10) % len(templates)
    return templates[idx]


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run(base_url: str, course_id: str, file_id: str, pdf_path: str, question: str) -> None:
    _header("Socratic Agent Simulator — RAG Backend Validation")
    print(f"  URL:      {base_url}")
    print(f"  Course:   {course_id}")
    print(f"  Question: {question}")

    # ------------------------------------------------------------------ Step 1
    _step(1, "Ensure file is indexed")
    t0 = time.monotonic()

    if not file_id:
        if not pdf_path:
            print("ERROR: provide --file <file_id> or --pdf <path>")
            sys.exit(1)
        file_id = upload_pdf(base_url, course_id, pdf_path)

    file_info = poll_until_indexed(base_url, file_id)
    elapsed = time.monotonic() - t0

    _ok(
        f"Indexed in {elapsed:.1f}s — "
        f"chunks={file_info.get('chunk_count')}  "
        f"pages={file_info.get('total_page_count')}  "
        f"empty_pages={len(file_info.get('empty_pages') or [])}  "
        f"ocr_completed={file_info.get('ocr_completed')}"
    )

    # ------------------------------------------------------------------ Step 2
    _step(2, f"Retrieve — query: {question!r}")
    t1 = time.monotonic()
    initial_response = retrieve(base_url, course_id, question)
    retrieval_time = time.monotonic() - t1

    initial_results = initial_response.get("results", [])
    has_unindexed   = initial_response.get("has_unindexed_pages", False)
    classification  = classify_results(initial_results)

    _info(f"Retrieval time:     {retrieval_time:.2f}s")
    _info(f"Results:            {_scores_summary(initial_results)}")
    _info(f"Classification:     {classification.upper()}")
    _info(f"has_unindexed_pages:{has_unindexed}")

    if initial_results:
        print()
        for i, r in enumerate(initial_results[:3]):
            conf = "LOW" if r.get("low_confidence") else "ok"
            print(
                f"    #{i+1}  score={r['score']:.3f} [{conf}]  "
                f"page={r.get('page','?')}  "
                f"{r['text'][:80].replace(chr(10),' ')!r}"
            )

    # ------------------------------------------------------------------ Step 3
    _step(3, "Agent decision")

    if classification == "strong":
        # ---- CASE A -------------------------------------------------------
        top_chunk = initial_results[0]
        socratic_q = make_socratic_question(top_chunk)
        _ok("CASE A — strong results found")
        _info(f"Top chunk: page={top_chunk.get('page')}  score={top_chunk['score']:.3f}")
        print()
        print("  🎓 Socratic question generated:")
        print(f"     {socratic_q}")

    elif classification in ("weak", "empty") and has_unindexed:
        # ---- CASE B -------------------------------------------------------
        _warn("CASE B — weak results + unindexed pages detected → triggering OCR")

        page_num = _extract_page_number(question)
        if not page_num:
            _warn("Could not extract page number from question — scanning all empty pages")
            page_num = (file_info.get("empty_pages") or [None])[0]

        if not page_num:
            _warn("No page number available — falling back to CASE C")
            print("\n  🤔 Clarification: Which section or topic are you referring to?")
            return

        _info(f"Extracted page number: {page_num}")
        _info(f"Triggering OCR for page {page_num} ...")

        t2 = time.monotonic()
        ocr_result = trigger_page_ocr(base_url, file_id, page_num)
        ocr_time = time.monotonic() - t2

        indexed = ocr_result.get("indexed", False)
        text_len = ocr_result.get("ocr_text_length", 0)
        _ok(f"OCR done in {ocr_time:.1f}s — indexed={indexed}  text_length={text_len}")

        if not indexed:
            _warn("OCR returned no text for this page")
            print("\n  🤔 Clarification: Which section or topic are you referring to?")
            return

        # Re-retrieve after OCR
        _info("Re-retrieving after OCR ...")
        t3 = time.monotonic()
        post_response = retrieve(base_url, course_id, question)
        post_retrieval_time = time.monotonic() - t3

        post_results = post_response.get("results", [])
        post_classification = classify_results(post_results)

        # Compare
        before_top = initial_results[0]["score"] if initial_results else 0.0
        after_top  = post_results[0]["score"] if post_results else 0.0
        improvement = after_top - before_top

        print()
        print("  📊 Before vs After OCR:")
        print(f"     Before: {_scores_summary(initial_results)}")
        print(f"     After:  {_scores_summary(post_results)}")
        print(f"     Improvement: {improvement:+.3f}  ({post_classification.upper()})")

        if post_classification == "strong":
            top_chunk = post_results[0]
            socratic_q = make_socratic_question(top_chunk)
            print()
            _ok("OCR improved retrieval to STRONG")
            print("  🎓 Socratic question generated:")
            print(f"     {socratic_q}")
        else:
            _warn(f"Results still {post_classification.upper()} after OCR")
            print("\n  🤔 Clarification: Which section or topic are you referring to?")

    else:
        # ---- CASE C -------------------------------------------------------
        _warn("CASE C — weak results, no OCR available")
        print("\n  🤔 Clarification: Which section or topic are you referring to?")

    # ------------------------------------------------------------------ Summary
    total = time.monotonic() - t0
    _header(f"Done — total time: {total:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Socratic RAG simulator")
    parser.add_argument("--url",      default=DEFAULT_BASE_URL,  help="API base URL")
    parser.add_argument("--course",   default=DEFAULT_COURSE_ID, help="Course UUID")
    parser.add_argument("--file",     default=DEFAULT_FILE_ID,   help="Existing file UUID (skip upload)")
    parser.add_argument("--pdf",      default=DEFAULT_PDF_PATH,  help="PDF path to upload")
    parser.add_argument("--question", default=DEFAULT_QUESTION,  help="Simulated student question")
    args = parser.parse_args()

    if not args.course:
        print("ERROR: --course <course_id> is required")
        sys.exit(1)
    if not args.file and not args.pdf:
        print("ERROR: provide --file <file_id> OR --pdf <path>")
        sys.exit(1)

    run(
        base_url=args.url.rstrip("/"),
        course_id=args.course,
        file_id=args.file,
        pdf_path=args.pdf,
        question=args.question,
    )
