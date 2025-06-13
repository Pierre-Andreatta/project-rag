from pydantic import BaseModel
from typing import Optional
from enum import Enum


class DocumentDomain(BaseModel):
    id: int
    content: str
    similarity: float
    source_id: Optional[int] = None

    class Config:
        exclude_none = True


class SourceTypeEnum(str, Enum):
    # HACK: Domain class used in Data layer
    DEFAULT = 'DEFAULT'
    WEB = "web"
    PDF = "pdf"
    YOUTUBE = "youtube"


class RejectReasonEnum(str, Enum):
    # HACK: Domain class used in Data layer
    INAPPROPRIATE = "inappropriate"
    DUPLICATE = "duplicated"
    LOW_QUALITY = "low_quality"
    OUTDATED = "obsolete"


class LanguageEnum(str, Enum):
    FR = "fr"
    EN = "en"
