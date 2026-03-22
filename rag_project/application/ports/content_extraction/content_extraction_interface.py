from abc import ABC, abstractmethod

from rag_project.domain.enums import SourceTypeEnum


class ContentExtractorInterface(ABC):
    @abstractmethod
    def extract(self, source_path: str) -> str:
        pass


class ContentExtractorFactoryInterface(ABC):
    @abstractmethod
    def get_extractor(self, source_type: SourceTypeEnum) -> ContentExtractorInterface:
        pass
