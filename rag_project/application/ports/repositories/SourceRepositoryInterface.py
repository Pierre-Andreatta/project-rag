from abc import ABC, abstractmethod
from typing import List, Dict

from rag_project.domain.enums import SourceTypeEnum


class SourceRepositoryInterface(ABC):

    @abstractmethod
    def create_source(self, source_path: str, source_type: SourceTypeEnum = SourceTypeEnum.DEFAULT):
        pass

    @abstractmethod
    def get_or_create_source(self, source_path: str, source_type: SourceTypeEnum = SourceTypeEnum.DEFAULT):
        pass

    @abstractmethod
    def get_source_by_path_to_content(self, path: str):
        pass

    @abstractmethod
    def get_source_by_id(self, source_id: int):
        pass

    @abstractmethod
    def approve_source(self, source_id: int):
        pass

    @abstractmethod
    def reject_source(self, source_id: int, reason: int):
        pass

    @abstractmethod
    def list_sources(
            self,
            *,
            only_accepted: bool = None,
            source_type: SourceTypeEnum = None,
            limit: int = 100
    ):
        pass
