from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel


class FileUploadResponse(BaseModel):
    file_id: UUID
    course_id: UUID
    status: str
    message: str


class FileStatusResponse(BaseModel):
    file_id: UUID
    course_id: UUID
    filename: str
    status: str
    total_page_count: Optional[int]
    extractable_page_count: Optional[int]
    chunk_count: Optional[int]
    error_message: Optional[str]
    uploaded_at: datetime
    indexed_at: Optional[datetime]

    model_config = {"from_attributes": True}

    # Pydantic doesn't know that ORM field `id` maps to `file_id`.
    # We handle this with a validator below.
    @classmethod
    def from_orm_file(cls, f) -> "FileStatusResponse":
        return cls(
            file_id=f.id,
            course_id=f.course_id,
            filename=f.filename,
            status=f.status,
            total_page_count=f.total_page_count,
            extractable_page_count=f.extractable_page_count,
            chunk_count=f.chunk_count,
            error_message=f.error_message,
            uploaded_at=f.uploaded_at,
            indexed_at=f.indexed_at,
        )


class FileDeleteResponse(BaseModel):
    file_id: UUID
    deleted: bool


class CourseFilesResponse(BaseModel):
    course_id: UUID
    files: List[FileStatusResponse]
