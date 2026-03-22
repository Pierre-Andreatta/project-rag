from abc import ABC, abstractmethod
from typing import List, Dict

from rag_project.domain.enums import SourceTypeEnum


class ContentRepositoryInterface(ABC):

    @abstractmethod
    def store_chunks(
            self,
            chunks: List[str],
            embeddings: List[List[float]],
            source_path: str,
            source_type: SourceTypeEnum = SourceTypeEnum.DEFAULT
    ) -> int:
        pass

    @abstractmethod
    def find_similar_contents(
            self,
            query_vector: List[float],
            top_k: int,
            min_similarity: float
    ) -> List:
        pass

    @abstractmethod
    def bulk_insert(self, contents: List[Dict], source_id: int) -> int:
        pass
