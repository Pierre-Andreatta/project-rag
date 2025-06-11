from typing import List, Dict
from rag_project.db.models.content import ContentORM
from rag_project.db.models.source import SourceORM
from rag_project.db.crud.base_crud import BaseCRUD
from rag_project.domain.models import SourceTypeEnum
from rag_project.db.crud.source import SourceCRUD


class ContentCRUD(BaseCRUD):

    def __init__(self, session):
        super().__init__(session)
        self.source_crud = SourceCRUD(session)

    def store_chunks(
            self,
            chunks: List[str],
            embeddings: List[List[float]],
            source_url: str,
            source_type: SourceTypeEnum = None
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

    def bulk_insert(self, contents: List[Dict], source_id: int) -> int:
        # Massive Insert
        db_contents = [
            ContentORM(
                content=c['text'],
                embedding=c['embedding'],
                source_id=source_id
            ) for c in contents
        ]
        self.session.bulk_save_objects(db_contents)
        return len(db_contents)
