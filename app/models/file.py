import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, String, Text
from app.database import GUID
from sqlalchemy.orm import relationship

from app.database import Base


class FileStatus(str, enum.Enum):
    # str mixin allows direct string comparison and JSON serialisation without
    # calling .value. Stored as String (not SQLAlchemy Enum) to avoid
    # CREATE TYPE DDL and migration friction when adding values.
    uploaded = "uploaded"
    processing = "processing"
    indexed = "indexed"
    failed = "failed"


class File(Base):
    __tablename__ = "files"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    course_id = Column(
        GUID(),
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = Column(Text, nullable=False)
    storage_path = Column(Text, nullable=False)
    size_bytes = Column(BigInteger, nullable=False)

    # total_page_count: every page in the PDF, including blank/scanned pages.
    # extractable_page_count: pages that yielded actual text via PyMuPDF.
    # Both are NULL until the ingestion task populates them.
    total_page_count = Column(Integer, nullable=True)
    extractable_page_count = Column(Integer, nullable=True)

    # Stored as String, not Enum. See FileStatus note above.
    status = Column(String, nullable=False, default=FileStatus.uploaded, index=True)
    error_message = Column(Text, nullable=True)
    chunk_count = Column(Integer, nullable=False, default=0)

    uploaded_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    indexed_at = Column(DateTime(timezone=True), nullable=True)

    course = relationship("Course", back_populates="files")
    chunks = relationship("Chunk", back_populates="file", cascade="all, delete-orphan")
