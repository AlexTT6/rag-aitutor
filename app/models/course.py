import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, Text
from app.database import GUID
from sqlalchemy.orm import relationship

from app.database import Base


class Course(Base):
    __tablename__ = "courses"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    files = relationship("File", back_populates="course", cascade="all, delete-orphan")
