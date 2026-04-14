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
        db.commit()
        t0 = time.monotonic()
        ocr_cache_path = file.storage_path.replace(".pdf", "_ocr_cache.json")
        extraction = extract_pages(
            file.storage_path,
            ocr_cache_path=ocr_cache_path,
            ocr_enabled=settings.OCR_ENABLED,
        )
        file.total_page_count = extraction.total_page_count
        file.extractable_page_count = extraction.extractable_page_count
        db.commit()
        logger.info(
            f"[ingestion] extract done file_id={file_id} "
            f"pages={extraction.total_page_count} "
            f"extractable={extraction.extractable_page_count} "
            f"ocr_pages={extraction.ocr_page_count} "
            f"ocr_enabled={settings.OCR_ENABLED} "
            f"elapsed={time.monotonic()-t0:.1f}s"
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

        # Phase: chunk
        if extraction.ocr_page_count > 0:
            file.status = FileStatus.ocr
            db.commit()
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
