from typing import List

from sentence_transformers import SentenceTransformer

from rag_project.application.ports.embeddings.embedding_interface import EmbeddingInterface
from rag_project.exceptions import ValidationError, IngestionError


class SentenceTransformerEmbeddingAdapter(EmbeddingInterface):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

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
