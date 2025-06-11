from pydantic import BaseModel
from typing import Any
from enum import Enum


class DocumentDomain(BaseModel):
    id: int
    content: str
    embedding: list[float]
    meta: dict[str, Any]


class SourceTypeEnum(str, Enum):
    # HACK: Domain class used in Data
    DEFAULT = 'DEFAULT'
    WEB = "web"
    PDF = "pdf"
    YOUTUBE = "youtube"


class RejectReasonEnum(str, Enum):
    # HACK: Domain class used in Data
    INAPPROPRIATE = "inappropriate"
    DUPLICATE = "duplicated"
    LOW_QUALITY = "low_quality"
    OUTDATED = "obsolete"
