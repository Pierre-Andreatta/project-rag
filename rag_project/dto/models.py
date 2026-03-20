from pydantic import BaseModel, Field
from typing import Optional, List

from rag_project.domain.enums import SourceTypeEnum, IngestionStatus, LanguageEnum


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


class IngestionRequest(BaseModel):
    source_type: SourceTypeEnum = Field(..., description="Type of source to ingest")
    source_path: str = Field(..., description="Path or URL to the source")
    user_id: Optional[str] = Field(None, description="User ID for multi-tenant support")

    class Config:
        schema_extra = {
            "example": {
                "source_type": "pdf",
                "source_path": "/path/to/document.pdf"
            }
        }


class IngestionResponse(BaseModel):
    chunks_stored: int = Field(..., description="Number of chunks stored")
    status: IngestionStatus = Field(..., description="Status of the ingestion")
    source_id: Optional[str] = Field(None, description="ID of the created source")

    class Config:
        schema_extra = {
            "example": {
                "chunks_stored": 42,
                "status": "success",
                "source_id": "uuid-123-456"
            }
        }


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    min_k: Optional[int] = Field(default=1, ge=1, le=10)
    language: Optional[LanguageEnum] = Field(default=LanguageEnum.FR)
    llm_model: Optional[str] = Field(default="gpt-3.5-turbo")
    min_similarity: Optional[float] = Field(default=0.4, ge=0.0, le=1.0)


class SourceInfo(BaseModel):
    id: int
    path: str
    type: str


class ChatResponse(BaseModel):
    response: str
    sources: List[SourceInfo]  # TODO: look if already exist

    class Config:
        json_encoders = {
            LanguageEnum: lambda v: v.value
        }
