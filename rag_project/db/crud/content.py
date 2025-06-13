from typing import List, Dict
from sqlalchemy import cast, literal, func
from sqlalchemy.exc import SQLAlchemyError

from rag_project.db.models.content import ContentORM, Vector
from rag_project.db.crud.base_crud import BaseCRUD
from rag_project.domain.models import SourceTypeEnum
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
            source_url: str,
            source_type: SourceTypeEnum = SourceTypeEnum.DEFAULT
    ) -> int:

        source = self.source_crud.get_or_create_source(source_url, source_type)

        contents = [
            ContentORM(
                content=text,
                embedding=emb,
                source_id=source.id
            )
            for text, emb in zip(chunks, embeddings)
        ]

        self.session.bulk_save_objects(contents)  # FIXME: check if need bulk_save_objects
        return len(contents)

    def find_similar_contents(
            self,
            query_vector: List[float],
            top_k: int = 5,
            min_similarity: float = 0.7
    ) -> List[dict]:
        # FIXME: HINT: No function matches the given name and argument types. You might need to add explicit type casts.
        # embedding_cast = cast(literal(query_vector), Vector(384))
        # distance_op = ContentORM.embedding.cosine_distance(embedding_cast)
        distance_op = func.cosine_distance(ContentORM.embedding, query_vector)

        results = (
            self.session.query(
                ContentORM,
                distance_op.label("distance")
            )
            .filter(distance_op < (1 - min_similarity))
            .order_by(distance_op)
            .limit(top_k)
            .all()
        )

        return [{
            'id': content.id,
            'content': content.content,
            'similarity': 1 - distance,  # Convert distance to similarity
            'source': content.source.path_to_content if content.source else None
        } for content, distance in results]

    def bulk_insert(self, contents: List[Dict], source_id: int) -> int:
        # Massive Insert
        contents = [
            ContentORM(
                content=c['text'],
                embedding=c['embedding'],
                source_id=source_id
            ) for c in contents
        ]
        self.session.bulk_save_objects(contents)
        return len(contents)
