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
    file_exists: bool = True   # False when the PDF was lost from disk (e.g. after redeploy)

    # OCR progress — only non-None while status="ocr".
    # ocr_pages_total: pages queued for Vision OCR this run.
    # ocr_pages_done:  pages completed so far (success or failure).
    ocr_pages_total: Optional[int] = None
    ocr_pages_done: Optional[int] = None

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_file(cls, f) -> "FileStatusResponse":
        import os
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
            file_exists=os.path.exists(f.storage_path),
            ocr_pages_total=getattr(f, "ocr_pages_total", None),
            ocr_pages_done=getattr(f, "ocr_pages_done", None),
        )


class FileDeleteResponse(BaseModel):
    file_id: UUID
    deleted: bool


class CourseFilesResponse(BaseModel):
    course_id: UUID
    files: List[FileStatusResponse]
