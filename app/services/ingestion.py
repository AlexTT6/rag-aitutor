import logging
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

# cl100k_base is the tokeniser used by text-embedding-3-small and GPT-4.
# For the local model (all-MiniLM-L6-v2) this is an approximation, but it
# is far more accurate than a naive word split and cheap to compute.
_tokenizer = tiktoken.get_encoding("cl100k_base")


def run_ingestion(file_id: str, db: Session) -> None:
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        logger.warning(f"[ingestion] file_id={file_id} not found in DB — skipping")
        return

    logger.info(f"[ingestion] START file_id={file_id} filename={file.filename}")

    # Phase 1 — mark processing
    file.status = FileStatus.processing
    db.commit()

    qdrant_vectors_written = False

    try:
        # Phase 2 — extract text from PDF
        logger.info(f"[ingestion] extracting pages file_id={file_id}")
        extraction = extract_pages(file.storage_path)
        file.total_page_count = extraction.total_page_count
        file.extractable_page_count = extraction.extractable_page_count
        db.commit()
        logger.info(
            f"[ingestion] extracted file_id={file_id} "
            f"total_pages={extraction.total_page_count} "
            f"extractable={extraction.extractable_page_count} "
            f"ocr_pages={extraction.ocr_page_count}"
        )

        # Phase 3 — enforce page limit
        if extraction.total_page_count > settings.MAX_PAGES:
            raise ValueError(
                f"PDF has {extraction.total_page_count} pages; "
                f"maximum allowed is {settings.MAX_PAGES}."
            )

        # Phase 4 — reject PDFs with no extractable content at all
        # (OCR already ran on image pages, so this only fires if everything failed)
        if extraction.extractable_page_count == 0:
            raise ValueError(
                "No text could be extracted from this PDF — "
                "neither native text nor OCR produced any content."
            )

        # Phase 5 — chunk text
        logger.info(f"[ingestion] chunking file_id={file_id}")
        chunks = chunk_document(extraction.pages)
        if not chunks:
            raise ValueError(
                "No text chunks could be produced from this PDF. "
                "It may contain only images or non-text content."
            )
        logger.info(f"[ingestion] chunked file_id={file_id} chunks={len(chunks)}")

        # Phase 6 — embed chunks
        logger.info(f"[ingestion] embedding file_id={file_id} chunks={len(chunks)}")
        try:
            texts = [c.text for c in chunks]
            embeddings = get_embeddings(texts)
        except Exception as embed_err:
            raise ValueError(f"Embedding failed: {embed_err}") from embed_err
        logger.info(f"[ingestion] embedded file_id={file_id}")

        # Phase 7 — write to Qdrant
        logger.info(f"[ingestion] writing to Qdrant file_id={file_id}")
        qdrant_ids = insert_chunks(
            file_id=file_id,
            course_id=str(file.course_id),
            chunks=chunks,
            embeddings=embeddings,
            filename=file.filename or "",
        )
        qdrant_vectors_written = True
        logger.info(f"[ingestion] Qdrant done file_id={file_id} vectors={len(qdrant_ids)}")

        # Phase 8 — write chunks to Postgres
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

        # Phase 9 — mark indexed
        file.status = FileStatus.indexed
        file.chunk_count = len(db_chunks)
        file.indexed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(
            f"[ingestion] INDEXED file_id={file_id} "
            f"chunks={len(db_chunks)}"
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

        # Re-fetch after rollback — ORM object is detached after rollback
        file = db.query(File).filter(File.id == file_id).first()
        if file:
            full_msg = str(e)
            # Preserve start and end of long error messages
            if len(full_msg) > 500:
                error_message = full_msg[:250] + " ... " + full_msg[-245:]
            else:
                error_message = full_msg
            file.status = FileStatus.failed
            file.error_message = error_message
            db.commit()
