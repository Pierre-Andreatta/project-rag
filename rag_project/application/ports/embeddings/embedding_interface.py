from abc import ABC, abstractmethod
from typing import List

from rag_project.exceptions import ValidationError, IngestionError


class EmbeddingInterface(ABC):

    @abstractmethod
    def __init__(self, model: str):
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass

    def embed_single(self, text: str) -> List[float]:
        """Used to embed a single text"""
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty")

        if len(text.strip()) < 5:
            raise ValidationError(f"Text '{text}' is too short (minimum 5 characters)")

        embeddings = self.embed([text])
        return embeddings[0]
