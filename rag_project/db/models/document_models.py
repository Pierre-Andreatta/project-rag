import json

from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.types import UserDefinedType
from sqlalchemy.sql import func
from rag_project.db.base import Base


class Vector(UserDefinedType):
    cache_ok = True

    def __init__(self, dim=384):
        self.dim = dim

    def get_col_spec(self):
        return f"VECTOR({self.dim})"

    def bind_processor(self, dialect):
        def process(value):
            if isinstance(value, list):
                return f"[{','.join(str(v) for v in value)}]"
            return value
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if isinstance(value, str):
                return json.loads(value)
            return value
        return process


class DocumentORM(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String)
    meta = Column(JSON, nullable=True)
    embedding = Column(Vector(384))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
