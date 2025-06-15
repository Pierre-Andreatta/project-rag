import json
from sqlalchemy import Column, Integer, DateTime, Text, ForeignKey, Index, Float
from sqlalchemy.orm import relationship
from sqlalchemy.types import UserDefinedType
from sqlalchemy.sql import func
from rag_project.db.base import Base


class Vector(UserDefinedType):
    cache_ok = True

    def __init__(self, dim=384):
        self.dim = dim

    def get_col_spec(self) -> str:
        return f"VECTOR({self.dim})"


class ContentORM(Base):
    __tablename__ = 'contents'
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embedding = Column(Vector(384))
    source_id = Column(Integer, ForeignKey('sources.id'))
    created_at = Column(DateTime, server_default=func.now())
    last_accessed = Column(DateTime)

    def cosine_distance(self, other):
        return func.cosine_distance(self.embedding, other)

    __table_args__ = (
        Index('ix_embedding_cosine', embedding,
              postgresql_using='ivfflat',
              postgresql_with={'lists': 10},  # Increase according to the table size
              postgresql_ops={'embedding': 'vector_cosine_ops'}),  # Optimized for all-MiniLM-L6-v2
    )
