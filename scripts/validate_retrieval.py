"""
Retrieval validation script — read-only, nothing is written.

Usage (run from project root):
    python -m scripts.validate_retrieval \
        --course_id <course_id> \
        --top_k 5 \
        --threshold 0.35

Then enter queries interactively, or pass them via --queries:
    python -m scripts.validate_retrieval \
        --course_id 3 \
        --queries "what is the IVT theorem?" "define continuity" "prove MVT"

For each query it prints:
  - raw top_k hits (id, score, 200-char preview)
  - how many survive the threshold filter
  - per-hit: PASS or FAIL vs threshold, and whether it would be in final results

To check whether an expected answer appears in the results, pass --expected:
    --expected "Intermediate Value Theorem"

The script will highlight any result chunk whose text contains the expected string.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.embedder import embed_query
from app.services.vector_store import search_chunks

SEP = "-" * 60


def validate_query(
    query: str,
    course_id: str,
    top_k: int,
    threshold: float,
    expected: str | None,
) -> None:
    print(f"\n{'='*60}")
    print(f"QUERY : {query!r}")
    print(f"  course_id  : {course_id}")
    print(f"  top_k      : {top_k}")
    print(f"  threshold  : {threshold}")
    if expected:
        print(f"  expected   : {expected!r}")
    print(SEP)

    # ── Embed ─────────────────────────────────────────────────────
    vector = embed_query(query)
    if not vector:
        print("  ERROR: embedding returned empty vector")
        return

    # ── Raw hits ──────────────────────────────────────────────────
    hits = search_chunks(vector, course_id, top_k)

    print(f"RAW HITS: {len(hits)} returned by Qdrant (top_k={top_k})")
    print()

    passed  = 0
    expected_in_raw     = False
    expected_in_results = False

    for rank, hit in enumerate(hits, start=1):
        payload  = hit.payload or {}
        text     = payload.get("text", "")
        above    = hit.score >= threshold
        contains = expected and expected.lower() in text.lower()

        if above:
            passed += 1
        if contains:
            expected_in_raw = True
            if above:
                expected_in_results = True

        status   = "PASS" if above else "FAIL (below threshold)"
        marker   = " ◄ EXPECTED MATCH" if contains else ""

        print(f"  [{rank}] id={hit.id}")
        print(f"       score   : {hit.score:.4f}  →  {status}{marker}")
        print(f"       page    : {payload.get('page', '?')}")
        print(f"       preview : {repr(text[:200])}")
        print()

    # ── Summary ───────────────────────────────────────────────────
    print(SEP)
    print(f"SUMMARY")
    print(f"  raw hits          : {len(hits)}")
    print(f"  above threshold   : {passed}")
    print(f"  dropped           : {len(hits) - passed}")

    if expected:
        print(f"  expected in raw   : {'YES' if expected_in_raw else 'NO  ← not in top-' + str(top_k)}")
        print(f"  expected in final : {'YES' if expected_in_results else 'NO  ← below threshold or not retrieved'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate retrieval pipeline")
    parser.add_argument("--course_id",  required=True,        help="Course ID to query against")
    parser.add_argument("--top_k",      type=int, default=5,  help="Number of candidates to retrieve (default: 5)")
    parser.add_argument("--threshold",  type=float,           help="Score threshold (default: from config)")
    parser.add_argument("--queries",    nargs="*",            help="One or more query strings")
    parser.add_argument("--expected",                         help="Substring to look for in results (for answer-location check)")
    args = parser.parse_args()

    threshold = args.threshold if args.threshold is not None else settings.RETRIEVAL_SCORE_THRESHOLD

    print(f"\nRETRIEVAL VALIDATION")
    print(f"  course_id  : {args.course_id}")
    print(f"  top_k      : {args.top_k}")
    print(f"  threshold  : {threshold}  (config default: {settings.RETRIEVAL_SCORE_THRESHOLD})")
    print(f"  collection : {settings.QDRANT_COLLECTION}")

    queries = args.queries or []

    if not queries:
        print("\nNo --queries provided. Enter queries interactively (empty line to quit):\n")
        while True:
            try:
                q = input("  query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            queries.append(q)

    for query in queries:
        validate_query(
            query=query,
            course_id=args.course_id,
            top_k=args.top_k,
            threshold=threshold,
            expected=args.expected,
        )

    print(f"\n{'='*60}")
    print(f"DONE — {len(queries)} query/queries validated")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
