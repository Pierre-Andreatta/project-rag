from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Enum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from rag_project.db.base import Base
from rag_project.domain.models import SourceTypeEnum, RejectReasonEnum


class RejectReasonORM(Base):
    __tablename__ = 'reject_reasons'
    id = Column(Integer, primary_key=True)
    reason = Column(Enum(RejectReasonEnum), unique=True, nullable=False)
    severity = Column(Integer, default=3, nullable=False)

    # Relation
    rejected_sources = relationship("SourceORM", back_populates="rejection_reason")


class CategoryORM(Base):
    # Category or Tag that user can apply to a source
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    color = Column(String(7), default='#b7bec7')
    is_system = Column(Boolean, default=False)

    # Relation
    sources = relationship(
        "SourceORM",
        secondary="source_categories",
        back_populates="categories"  # Bidirectional
    )


class SourceORM(Base):
    __tablename__ = 'sources'
    id = Column(Integer, primary_key=True)
    path_to_content = Column(String(500), unique=True, nullable=False)
    source_type = Column(Enum(SourceTypeEnum), nullable=False)
    is_accepted = Column(Boolean, default=True, nullable=False)
    rejection_reason = Column(Enum(RejectReasonEnum), ForeignKey('reject_reasons.reason'), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relations
    rejection_reason_obj = relationship(
        "RejectReasonORM",
        back_populates="rejected_sources"
    )

    categories = relationship(  # Many-to-many
        "CategoryORM",
        secondary="source_categories",
        back_populates="sources"  # Bidirectional
    )

    __table_args__ = (
        Index('idx_source_path', path_to_content),
        Index('idx_source_status', is_accepted),
        Index('idx_rejection_reason', rejection_reason),
        Index('idx_source_created_at', created_at.desc())
    )


# Association tables

class SourceCategoryORM(Base):
    __tablename__ = 'source_categories'
    source_id = Column(Integer, ForeignKey('sources.id', ondelete="CASCADE"), primary_key=True)
    category_id = Column(Integer, ForeignKey('categories.id', ondelete="CASCADE"), primary_key=True)
