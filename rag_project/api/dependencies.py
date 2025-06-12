from fastapi import Depends
from sentence_transformers import SentenceTransformer

from rag_project.db.session import get_session
from rag_project.services.ingestion_service import IngestionService


def get_ingestion_service(session=Depends(get_session)) -> IngestionService:
    return IngestionService(session_factory=lambda: session)


def get_embedding_model() -> SentenceTransformer:
    # Load model only once (implicit cache)
    return SentenceTransformer("all-MiniLM-L6-v2")
