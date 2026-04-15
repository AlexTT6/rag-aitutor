import logging
import time
from datetime import datetime, timezone

import tiktoken
from sqlalchemy.orm import Session

from app.config import settings
from app.models.chunk import Chunk as ChunkModel
from app.models.file import File, FileStatus
from app.services.chunker import chunk_document
from app.services.embedder import get_embeddings
from app.services.extractor import extract_pages
from app.services.vector_store import delete_by_file_id, insert_chunks

logger = logging.getLogger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")


def run_ingestion(file_id: str, db: Session) -> None:
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        logger.warning(f"[ingestion] file_id={file_id} not found in DB — skipping")
        return

    logger.info(f"[ingestion] START file_id={file_id} filename={file.filename}")
    t_start = time.monotonic()

    file.status = FileStatus.processing
    db.commit()

    qdrant_vectors_written = False

    try:
        # Phase: extract text from PDF (+ OCR if enabled)
        file.status = FileStatus.extracting
        file.ocr_pages_done = None
        file.ocr_pages_total = None
        db.commit()
        t0 = time.monotonic()
        ocr_cache_path = file.storage_path.replace(".pdf", "_ocr_cache.json")

        def _ocr_progress(pages_done: int, pages_total: int) -> None:
            """
            Called by extract_pages when OCR starts (pages_done=0) and after
            each page completes. Runs in the ingestion thread — DB access is safe.
            """
            if pages_done == 0:
                # OCR is starting: flip status and record total.
                file.status = FileStatus.ocr
                file.ocr_pages_total = pages_total
                file.ocr_pages_done = 0
                logger.info(
                    f"[ingestion] OCR starting file_id={file_id} "
                    f"pages_to_ocr={pages_total}"
                )
            else:
                file.ocr_pages_done = pages_done
            try:
                db.commit()
            except Exception as cb_err:
                logger.warning(f"[ingestion] progress commit failed: {cb_err}")
                db.rollback()

        extraction = extract_pages(
            file.storage_path,
            ocr_cache_path=ocr_cache_path,
            ocr_enabled=settings.OCR_ENABLED,
            progress_callback=_ocr_progress if settings.OCR_ENABLED else None,
        )
        file.total_page_count = extraction.total_page_count
        file.extractable_page_count = extraction.extractable_page_count
        # Clear OCR progress fields now that extraction is complete.
        file.ocr_pages_done = None
        file.ocr_pages_total = None
        db.commit()
        logger.info(
            f"[ingestion] extract done file_id={file_id} "
            f"pages={extraction.total_page_count} "
            f"extractable={extraction.extractable_page_count} "
            f"ocr_pages={extraction.ocr_page_count} "
            f"ocr_enabled={settings.OCR_ENABLED} "
            f"elapsed={time.monotonic()-t0:.1f}s"
        )
        _total_chars = sum(len(p.text) for p in extraction.pages)
        _ocr_chars = sum(len(p.text) for p in extraction.pages if p.ocr_used)
        logger.info(
            f"[ingestion:debug] FILE filename={file.filename!r} "
            f"total_pages={extraction.total_page_count}"
        )
        logger.info(
            f"[ingestion:debug] EXTRACTION "
            f"total_chars={_total_chars} "
            f"ocr_used={'yes' if extraction.ocr_page_count > 0 else 'no'} "
            f"ocr_chars={_ocr_chars}"
        )

        if extraction.total_page_count > settings.MAX_PAGES:
            raise ValueError(
                f"PDF has {extraction.total_page_count} pages; "
                f"maximum allowed is {settings.MAX_PAGES}."
            )

        if extraction.extractable_page_count == 0:
            raise ValueError(
                "No text could be extracted from this PDF — "
                "neither native text nor OCR produced any content."
            )

        # Phase: chunk (fast — usually <1 second)
        file.status = FileStatus.chunking
        db.commit()
        t0 = time.monotonic()
        chunks = chunk_document(extraction.pages)
        if not chunks:
            raise ValueError(
                "No text chunks could be produced from this PDF. "
                "It may contain only images or non-text content."
            )
        logger.info(
            f"[ingestion] chunk done file_id={file_id} "
            f"chunks={len(chunks)} "
            f"elapsed={time.monotonic()-t0:.1f}s"
        )
        _token_counts = [len(_tokenizer.encode(c.text)) for c in chunks]
        _avg_tokens = sum(_token_counts) / len(_token_counts)
        _sample_indices = [0, len(chunks) // 2, len(chunks) - 1]
        logger.info(
            f"[ingestion:debug] CHUNKING "
            f"total_chunks={len(chunks)} "
            f"avg_tokens={_avg_tokens:.0f}"
        )
        for _si in _sample_indices:
            logger.info(
                f"[ingestion:debug] CHUNK_SAMPLE idx={_si} "
                f"tokens={_token_counts[_si]} "
                f"text={chunks[_si].text[:200]!r}"
            )

        # Phase: embed
        file.status = FileStatus.embedding
        db.commit()
        t0 = time.monotonic()
        try:
            texts = [c.text for c in chunks]
            embeddings = get_embeddings(texts)
        except Exception as embed_err:
            raise ValueError(f"Embedding failed: {embed_err}") from embed_err
        logger.info(
            f"[ingestion] embed done file_id={file_id} "
            f"chunks={len(embeddings)} "
            f"elapsed={time.monotonic()-t0:.1f}s"
        )
        logger.info(
            f"[ingestion:debug] EMBEDDINGS "
            f"count={len(embeddings)} "
            f"model=text-embedding-3-small "
            f"dim={len(embeddings[0]) if embeddings else 0}"
        )

        # Phase: write to Qdrant
        t0 = time.monotonic()
        qdrant_ids = insert_chunks(
            file_id=file_id,
            course_id=str(file.course_id),
            chunks=chunks,
            embeddings=embeddings,
            filename=file.filename or "",
        )
        qdrant_vectors_written = True
        logger.info(
            f"[ingestion] qdrant done file_id={file_id} "
            f"vectors={len(qdrant_ids)} "
            f"elapsed={time.monotonic()-t0:.1f}s"
        )
        logger.info(
            f"[ingestion:debug] INDEXING "
            f"collection={settings.QDRANT_COLLECTION} "
            f"vectors_stored={len(qdrant_ids)} "
            f"status=ok"
        )

        # Phase: write chunks to Postgres
        db_chunks = [
            ChunkModel(
                file_id=file.id,
                course_id=file.course_id,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=len(_tokenizer.encode(chunk.text)),
                qdrant_id=qid,
            )
            for chunk, qid in zip(chunks, qdrant_ids)
        ]
        db.add_all(db_chunks)

        file.status = FileStatus.indexed
        file.chunk_count = len(db_chunks)
        file.indexed_at = datetime.now(timezone.utc)
        db.commit()

        logger.info(
            f"[ingestion] DONE file_id={file_id} "
            f"chunks={len(db_chunks)} "
            f"total_elapsed={time.monotonic()-t_start:.1f}s"
        )

    except Exception as e:
        logger.exception(f"[ingestion] FAILED file_id={file_id}: {e}")
        db.rollback()

        if qdrant_vectors_written:
            try:
                delete_by_file_id(file_id)
                logger.info(f"[ingestion] Qdrant cleanup done file_id={file_id}")
            except Exception as cleanup_err:
                logger.error(
                    f"[ingestion] Qdrant cleanup FAILED file_id={file_id}: {cleanup_err}"
                )

        file = db.query(File).filter(File.id == file_id).first()
        if file:
            full_msg = str(e)
            error_message = full_msg[:250] + " ... " + full_msg[-245:] if len(full_msg) > 500 else full_msg
            file.status = FileStatus.failed
            file.error_message = error_message
            db.commit()
