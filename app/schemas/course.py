from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class CourseCreateRequest(BaseModel):
    name: str


class CourseResponse(BaseModel):
    course_id: UUID
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}
