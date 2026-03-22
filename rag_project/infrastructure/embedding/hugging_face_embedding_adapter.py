from typing import List

from rag_project.application.ports.embeddings.embedding_interface import EmbeddingInterface


class HuggingFaceEmbeddingAdapter(EmbeddingInterface):

    def __init__(self, model):
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Cette méthode n'est pas encore implémentée")
