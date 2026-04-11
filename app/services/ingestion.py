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

# cl100k_base is the tokeniser used by text-embedding-3-small and GPT-4.
# For the local model (all-MiniLM-L6-v2) this is an approximation, but it
# is far more accurate than a naive word split and cheap to compute.
_tokenizer = tiktoken.get_encoding("cl100k_base")


def run_ingestion(file_id: str, db: Session) -> None:
    # file_id is a plain str and stays a str throughout.
    # delete_by_file_id always receives file_id directly, never an ORM attribute,
    # so it is safe to call after db.rollback() when ORM objects are detached.

    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        return

    # Phase 1
    file.status = FileStatus.processing
    db.commit()

    qdrant_vectors_written = False

    try:
        # Phase 2
        extraction = extract_pages(file.storage_path)
        file.total_page_count = extraction.total_page_count
        file.extractable_page_count = extraction.extractable_page_count
        db.commit()

        # Phase 3: enforce MAX_PAGES on total page count, not extractable count.
        if extraction.total_page_count > settings.MAX_PAGES:
            raise ValueError(
                f"PDF has {extraction.total_page_count} pages; "
                f"maximum allowed is {settings.MAX_PAGES}."
            )

        # Phase 4: reject PDFs where too few pages have extractable text.
        if extraction.total_page_count > 0:
            ratio = extraction.extractable_page_count / extraction.total_page_count
            if ratio < settings.MIN_EXTRACTABLE_RATIO:
                raise ValueError(
                    f"Only {extraction.extractable_page_count} of "
                    f"{extraction.total_page_count} pages contain extractable text "
                    f"({ratio:.0%}). PDF appears to be mostly scanned images. "
                    f"Minimum required: {settings.MIN_EXTRACTABLE_RATIO:.0%}."
                )

        # Phase 5
        chunks = chunk_document(extraction.pages)
        if not chunks:
            raise ValueError(
                "No text chunks could be produced from this PDF. "
                "It may contain only images or non-text content."
            )

        # Phase 6
        texts = [c.text for c in chunks]
        embeddings = get_embeddings(texts)

        # Phase 7: point of no return — Qdrant is written before Postgres.
        qdrant_ids = insert_chunks(
            file_id=file_id,
            course_id=str(file.course_id),
            chunks=chunks,
            embeddings=embeddings,
        )
        qdrant_vectors_written = True

        # Phase 8
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

        # Phase 9
        file.status = FileStatus.indexed
        file.chunk_count = len(db_chunks)
        file.indexed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as e:
        db.rollback()

        if qdrant_vectors_written:
            try:
                delete_by_file_id(file_id)
            except Exception as cleanup_error:
                print("Qdrant cleanup failed:", cleanup_error)

        # Re-fetch after rollback — the ORM object is detached after rollback.
        file = db.query(File).filter(File.id == file_id).first()
        if file:
            file.status = FileStatus.failed
            file.error_message = str(e)[:500]
            db.commit()
