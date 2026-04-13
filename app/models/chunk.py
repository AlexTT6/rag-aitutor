import uuid

from sqlalchemy import Column, ForeignKey, Integer, Text
from app.database import GUID
from sqlalchemy.orm import relationship

from app.database import Base


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    file_id = Column(
        GUID(),
        ForeignKey("files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # course_id is denormalized here to avoid a join during retrieval result hydration.
    course_id = Column(GUID(), nullable=False, index=True)
    page = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)   # global order within the file
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)    # naive word count, not real tokens

    # The UUID used as the point ID in Qdrant. Used to look up metadata
    # after a Qdrant search returns IDs.
    qdrant_id = Column(Text, nullable=False, unique=True)

    file = relationship("File", back_populates="chunks")
