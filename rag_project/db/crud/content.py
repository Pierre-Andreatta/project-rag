from typing import List, Dict
from sqlalchemy import cast, literal, func
from pgvector.sqlalchemy import Vector
from sqlalchemy.exc import SQLAlchemyError

from rag_project.db.models.content import ContentORM
from rag_project.db.crud.base_crud import BaseCRUD
from rag_project.dto.models import SourceTypeEnum, DocumentDto
from rag_project.db.crud.source import SourceCRUD
from rag_project.exceptions import DataBaseError
from rag_project.logger import get_logger

logger = get_logger(__name__)


class ContentCRUD(BaseCRUD):

    def __init__(self, session):
        super().__init__(session)
        self.source_crud = SourceCRUD(session)

    def store_chunks(
            self,
            chunks: List[str],
            embeddings: List[List[float]],
            source_path: str,
            source_type: SourceTypeEnum = SourceTypeEnum.DEFAULT
    ) -> int:

        if not chunks or not embeddings:
            raise DataBaseError("Chunks and embeddings cannot be empty")

        if len(chunks) != len(embeddings):
            raise DataBaseError(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")

        try:
            source = self.source_crud.get_or_create_source(source_path, source_type)

            contents = [
                ContentORM(
                    content=text,
                    embedding=emb,
                    source_id=source.id
                ) for text, emb in zip(chunks, embeddings)
            ]

            self.session.bulk_save_objects(contents)
            self.session.flush()

            logger.info(f"Successfully stored {len(contents)} chunks")
            return len(contents)

        except SQLAlchemyError as e:
            message = f"store_chunks: {str(e)}"
            raise DataBaseError(message) from e

    def find_similar_contents(
            self,
            query_vector: List[float],
            top_k: int,
            min_similarity: float
    ) -> List[DocumentDto]:
        try:
            casted_vector = cast(query_vector, Vector)
            distance_op = func.cosine_distance(ContentORM.embedding, casted_vector)
            similarity_op = literal(1.0) - distance_op  # Similarity direct calculation

            query = (
                self.session.query(
                    ContentORM,
                    similarity_op.label("similarity"),
                    distance_op.label("distance")
                )
                .filter(similarity_op >= min_similarity)  # More intuitif tahn distance
                .order_by(distance_op)
                .limit(top_k)
            )

            results = query.all()

            return [DocumentDto(
                id=content.id,
                content=content.content,
                similarity=float(similarity),
                source_data=self.source_crud.get_source_by_id(content.source_id)
            ) for content, similarity, _distance in results]
        except SQLAlchemyError as e:
            message = f"find_similar_contents: {e}"
            raise DataBaseError(message) from e

    def bulk_insert(self, contents: List[Dict], source_id: int) -> int:
        # Massive Insert
        contents = [
            ContentORM(
                content=c['text'],
                embedding=c['embedding'],
                source_id=source_id
            ) for c in contents
        ]
        try:
            self.session.bulk_save_objects(contents)
            return len(contents)
        except SQLAlchemyError as e:
            message = f"bulk_insert: {str(e)}"
            raise DataBaseError(message) from e
