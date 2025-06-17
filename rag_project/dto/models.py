from pydantic import BaseModel
from typing import Optional, List

from rag_project.domain.enums import SourceTypeEnum


class SourceDto(BaseModel):
    id: int
    source_path: Optional[str]
    source_type: Optional[SourceTypeEnum] = None

    class Config:
        exclude_none = True
        orm_mode = True


class DocumentDto(BaseModel):
    id: int
    content: str
    similarity: float
    source_data: Optional[SourceDto] = None

    class Config:
        exclude_none = True
        orm_mode = True


class AnswerDto(BaseModel):
    answer: str
    confidence: Optional[float] = None
    sources: Optional[List[SourceDto]] = None
