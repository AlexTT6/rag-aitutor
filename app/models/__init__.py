# Import all models here so Alembic's env.py can discover them
# via Base.metadata when generating migrations.
from app.database import Base
from app.models.course import Course
from app.models.file import File, FileStatus
from app.models.chunk import Chunk

__all__ = ["Base", "Course", "File", "FileStatus", "Chunk"]
