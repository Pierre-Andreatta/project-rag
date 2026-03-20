# adapters/storage_adapter.py
from abc import ABC, abstractmethod
from typing import List

from rag_project.db.crud.content import ContentCRUD
from rag_project.domain.enums import SourceTypeEnum


class StorageAdapter(ABC):
    @abstractmethod
    def store_chunks(self, chunks: List[str], embeddings: List[List[float]],
                     source_path: str, source_type: SourceTypeEnum) -> int:
        pass


class DatabaseStorageAdapter(StorageAdapter):
    def __init__(self, session):
        self.session = session

    def store_chunks(self, chunks: List[str], embeddings: List[List[float]],
                     source_path: str, source_type: SourceTypeEnum) -> int:
        content_crud = ContentCRUD(self.session)
        return content_crud.store_chunks(chunks, embeddings, source_path, source_type)
