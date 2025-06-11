import json
from sqlalchemy import Column, Integer, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
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


class ContentORM(Base):
    __tablename__ = 'contents'
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embedding = Column(Vector(384))
    source_id = Column(Integer, ForeignKey('sources.id'))
    created_at = Column(DateTime, server_default=func.now())
    last_accessed = Column(DateTime)

    source = relationship("Source", backref="contents")

    __table_args__ = (
        Index('ix_embedding_cosine', embedding,
              postgresql_using='ivfflat',
              postgresql_with={'lists': 10},  # Increase according to the table size
              postgresql_ops={'embedding': 'vector_cosine_ops'}),  # Optimized for all-MiniLM-L6-v2
    )
