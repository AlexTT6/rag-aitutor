"""
On-demand OCR service.

Two entry points:
  ocr_single_page(file_id, page_num, db)
    Synchronous. Renders one page, calls Vision API, chunks + embeds + indexes the result.
    Removes the page from file.empty_pages. Returns extracted text.
    Used when the agent knows exactly which page the student needs.

  ocr_missing_pages(file_id, db)
    Iterates file.empty_pages, calls ocr_single_page for each.
    Sets file.ocr_completed = True when done.
    Called when retrieval confidence is low and the file has unindexed pages.
    Designed to run in a background thread — caller should submit via executor.
"""

import concurrent.futures
import logging
import time

import fitz

from sqlalchemy.orm import Session

from app.models.chunk import Chunk as ChunkModel
from app.models.file import File, FileStatus
from app.services.chunker import chunk_document
from app.services.embedder import get_embeddings
from app.services.extractor import (
    OCR_HARD_DPI,
    OCR_MAX_WORKERS,
    OCR_MIN_CHARS,
    OCR_DPI,
    PageContent,
    _get_ocr_client,
    _is_good_text,
    _ocr_one,
    _render_page_b64,
    _load_ocr_cache,
    _save_ocr_cache,
)
from app.services.vector_store import delete_by_file_id, delete_by_qdrant_ids, insert_chunks

logger = logging.getLogger(__name__)


def ocr_single_page(file_id: str, page_num: int, db: Session) -> str:
    """
    OCR a single page and merge the result into the existing index.

    Steps:
      1. Render the page at appropriate DPI
      2. Call Vision API
      3. Chunk + embed the extracted text
      4. Upsert vectors to Qdrant (old vectors for this page are deleted first)
      5. Upsert chunk rows to Postgres
      6. Remove page_num from file.empty_pages

    Returns the extracted text (empty string if OCR produced nothing).
    Raises ValueError if file not found or PDF not on disk.
    """
    db_file = db.query(File).filter(File.id == file_id).first()
    if not db_file:
        raise ValueError(f"File {file_id} not found")

    import os
    if not os.path.exists(db_file.storage_path):
        raise ValueError(f"PDF not on disk: {db_file.storage_path}")

    t0 = time.monotonic()
    doc = fitz.open(db_file.storage_path)
    try:
        return _ocr_single_page_with_doc(doc, db_file, page_num, db, t0=t0)
    finally:
        doc.close()


def _render_and_check(fitz_page, page_num: int):
    """
    Main-thread-only (fitz is not thread-safe).
    Returns (b64, is_hard, native_text, detail) for a single page.
    """
    native_text = fitz_page.get_text("text").strip()
    is_hard = len(native_text) < OCR_MIN_CHARS
    dpi = OCR_HARD_DPI if is_hard else OCR_DPI
    b64 = _render_page_b64(fitz_page, dpi=dpi, quality=95 if is_hard else 85)
    detail = "high" if is_hard else "auto"
    return b64, is_hard, native_text, detail


def _write_ocr_result(
    db_file, page_num: int, native_text: str, ocr_text: str, db: Session
) -> int:
    """
    Main-thread-only DB writes for one page's OCR result.
    Handles: cache persist, delete old chunks, chunk + embed + insert, commit,
    and remove page from empty_pages.
    Returns number of chunks inserted (0 if OCR produced no text).
    Caller is responsible for try/except around embedding/indexing failures.
    """
    file_id = str(db_file.id)

    if not ocr_text.strip():
        logger.warning(f"[ocr_service] OCR returned empty for page={page_num}")
        _remove_from_empty_pages(db_file, page_num, db)
        db.commit()
        return 0

    # Persist OCR cache on disk
    ocr_cache_path = db_file.storage_path.replace(".pdf", "_ocr_cache.json")
    cache = _load_ocr_cache(ocr_cache_path)
    cache[page_num] = ocr_text
    _save_ocr_cache(ocr_cache_path, cache)

    # Combine native + OCR text
    combined = ocr_text if not native_text else f"{native_text}\n{ocr_text}"
    page_content = PageContent(page=page_num, text=combined, ocr_used=True)

    # Delete existing Postgres chunk rows for this page AND their Qdrant vectors,
    # then flush so the COUNT below sees the updated state.
    deleted_count = _delete_page_chunks(file_id, page_num, db)
    db.flush()

    chunks = chunk_document([page_content])
    if not chunks:
        _remove_from_empty_pages(db_file, page_num, db)
        db.commit()
        return 0

    texts = [c.text for c in chunks]
    embeddings = get_embeddings(texts)

    max_idx = db.query(ChunkModel).filter(
        ChunkModel.file_id == file_id
    ).count()
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = max_idx + i

    qdrant_ids = insert_chunks(
        file_id=file_id,
        course_id=str(db_file.course_id),
        chunks=chunks,
        embeddings=embeddings,
        filename=db_file.filename or "",
    )

    db_chunks = [
        ChunkModel(
            file_id=db_file.id,
            course_id=db_file.course_id,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            token_count=chunk.token_count,
            qdrant_id=qid,
        )
        for chunk, qid in zip(chunks, qdrant_ids)
    ]
    db.add_all(db_chunks)
    db_file.chunk_count = max(0, (db_file.chunk_count or 0) - deleted_count) + len(db_chunks)

    _remove_from_empty_pages(db_file, page_num, db)
    db.commit()
    return len(db_chunks)


def _ocr_single_page_with_doc(
    doc: fitz.Document, db_file, page_num: int, db: Session, *, t0: float = None
) -> str:
    """
    OCR one page using an already-open fitz.Document (single-page path).
    """
    if t0 is None:
        t0 = time.monotonic()
    file_id = str(db_file.id)

    total_pages = len(doc)
    if page_num < 1 or page_num > total_pages:
        raise ValueError(f"Page {page_num} out of range (1–{total_pages})")
    fitz_page = doc[page_num - 1]   # fitz is 0-indexed
    b64, is_hard, native_text, detail = _render_and_check(fitz_page, page_num)

    logger.info(
        f"[ocr_service] OCR_SINGLE file_id={file_id} page={page_num} "
        f"native_chars={len(native_text)} is_hard={is_hard}"
    )

    _, ocr_text = _ocr_one(page_num, b64, detail=detail)
    n_chunks = _write_ocr_result(db_file, page_num, native_text, ocr_text, db)

    logger.info(
        f"[ocr_service] OCR_SINGLE DONE file_id={file_id} page={page_num} "
        f"chunks={n_chunks} elapsed={time.monotonic()-t0:.1f}s"
    )
    return ocr_text


def ocr_missing_pages(file_id: str, db: Session) -> int:
    """
    OCR all pages listed in file.empty_pages, in parallel.

    Three phases:
      1. Render all pages sequentially (fitz is not thread-safe).
      2. Fan out OCR HTTP calls via ThreadPoolExecutor (OCR_MAX_WORKERS).
      3. Write results to DB sequentially in the main thread
         (SQLAlchemy Session is not thread-safe).

    Sets file.ocr_completed=True when done (even if some pages produced no text).
    Returns number of pages that produced text.
    Safe to call multiple times — empty_pages list shrinks as pages are processed.
    """
    db_file = db.query(File).filter(File.id == file_id).first()
    if not db_file:
        raise ValueError(f"File {file_id} not found")

    if db_file.ocr_completed:
        logger.info(f"[ocr_service] OCR already completed for file_id={file_id} — skipping")
        return 0

    pages = list(db_file.empty_pages or [])
    if not pages:
        db_file.ocr_completed = True
        db.commit()
        return 0

    import os
    if not os.path.exists(db_file.storage_path):
        raise ValueError(f"PDF not on disk: {db_file.storage_path}")

    t0 = time.monotonic()
    logger.info(
        f"[ocr_service] OCR_MISSING START file_id={file_id} pages={pages} "
        f"workers={OCR_MAX_WORKERS}"
    )

    # Phase 1 — render sequentially (fitz is not thread-safe).
    render_inputs = {}  # page_num -> (b64, is_hard, native_text, detail)
    doc = fitz.open(db_file.storage_path)
    try:
        total_pages = len(doc)
        for page_num in pages:
            if page_num < 1 or page_num > total_pages:
                logger.error(
                    f"[ocr_service] OCR_MISSING skip page={page_num} out of range (1–{total_pages})"
                )
                continue
            try:
                fitz_page = doc[page_num - 1]
                render_inputs[page_num] = _render_and_check(fitz_page, page_num)
            except Exception as e:
                logger.error(f"[ocr_service] OCR_MISSING render failed page={page_num}: {e}")
    finally:
        doc.close()

    t_render = time.monotonic() - t0

    # Phase 2 — OCR HTTP calls in parallel.
    ocr_results = {}  # page_num -> (native_text, ocr_text)
    t1 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS) as pool:
        future_to_page = {
            pool.submit(_ocr_one, page_num, b64, detail): page_num
            for page_num, (b64, _is_hard, _native, detail) in render_inputs.items()
        }
        for fut in concurrent.futures.as_completed(future_to_page):
            page_num = future_to_page[fut]
            native_text = render_inputs[page_num][2]
            try:
                _, ocr_text = fut.result()
                ocr_results[page_num] = (native_text, ocr_text)
            except Exception as e:
                logger.error(f"[ocr_service] OCR_MISSING ocr_one failed page={page_num}: {e}")
    t_ocr = time.monotonic() - t1

    # Phase 3 — DB writes sequentially on the main thread.
    t2 = time.monotonic()
    success_count = 0
    for page_num in pages:
        if page_num not in ocr_results:
            continue
        native_text, ocr_text = ocr_results[page_num]
        try:
            n_chunks = _write_ocr_result(db_file, page_num, native_text, ocr_text, db)
            if n_chunks > 0:
                success_count += 1
        except Exception as e:
            logger.error(f"[ocr_service] OCR_MISSING db-write failed page={page_num}: {e}")
            db.rollback()
    t_db = time.monotonic() - t2

    db_file = db.query(File).filter(File.id == file_id).first()
    if db_file:
        db_file.ocr_completed = True
        db.commit()

    logger.info(
        f"[ocr_service] OCR_MISSING DONE file_id={file_id} "
        f"processed={len(pages)} success={success_count} "
        f"render={t_render:.1f}s ocr={t_ocr:.1f}s db={t_db:.1f}s "
        f"total={time.monotonic()-t0:.1f}s"
    )
    return success_count


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_from_empty_pages(db_file: File, page_num: int, db: Session) -> None:
    from sqlalchemy.orm.attributes import flag_modified
    current = list(db_file.empty_pages or [])
    if page_num in current:
        current.remove(page_num)
    # Store None (not []) when empty — keeps has_unindexed_pages query correct
    db_file.empty_pages = current if current else None
    flag_modified(db_file, "empty_pages")
    # If all pages have been individually OCR'd, mark as completed
    if not current:
        db_file.ocr_completed = True


def _delete_page_chunks(file_id: str, page_num: int, db: Session) -> int:
    """
    Remove existing Postgres chunk rows and their Qdrant vectors for this page.
    Returns the number of rows deleted (used to adjust chunk_count).
    """
    rows = db.query(ChunkModel).filter(
        ChunkModel.file_id == file_id,
        ChunkModel.page == page_num,
    ).all()
    if not rows:
        return 0

    qdrant_ids = [r.qdrant_id for r in rows if r.qdrant_id]
    if qdrant_ids:
        try:
            delete_by_qdrant_ids(qdrant_ids)
        except Exception as e:
            logger.warning(f"[ocr_service] failed to delete Qdrant vectors for page={page_num}: {e}")

    for row in rows:
        db.delete(row)

    return len(rows)
