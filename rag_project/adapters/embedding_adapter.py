# adapters/embedding_adapter.py
from abc import ABC, abstractmethod
from typing import List

from rag_project.exceptions import ValidationError, IngestionError


class EmbeddingAdapter(ABC):
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


class SentenceTransformerEmbeddingAdapter(EmbeddingAdapter):
    def __init__(self, model):
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValidationError("No texts provided for embedding", field="texts")

        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True).tolist()
            if len(embeddings) != len(texts):
                raise IngestionError(f"Embedding count mismatch: {len(embeddings)} != {len(texts)}")
            return embeddings
        except Exception as e:
            raise IngestionError(f"Failed to embed texts: {str(e)}") from e


# Futurs adapters pour d'autres providers
class OpenAIEmbeddingAdapter(EmbeddingAdapter):
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Implementation avec OpenAI API
        pass


class HuggingFaceEmbeddingAdapter(EmbeddingAdapter):
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Implementation avec HuggingFace API
        pass
