from pydantic import BaseModel
from typing import Any


class DocumentDomain(BaseModel):
    id: int
    content: str
    embedding: list[float]
    meta: dict[str, Any]
