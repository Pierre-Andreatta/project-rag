from fastapi import Depends
from sentence_transformers import SentenceTransformer


from rag_project.application.use_cases.IngestionUseCase import IngestionUseCase
from rag_project.application.use_cases.RagUseCase import RagUseCase
from rag_project.infrastructure.db.session import get_session


def get_ingestion_service(session=Depends(get_session)) -> IngestionUseCase:
    # FIXME:
    return IngestionUseCase(session_factory=lambda: session)


def get_rag_service(session=Depends(get_session)) -> RagUseCase:
    return RagUseCase(session_factory=lambda: session)


def get_embedding_model() -> SentenceTransformer:
    # Load model only once (implicit cache)
    return SentenceTransformer("all-MiniLM-L6-v2")
