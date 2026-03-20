from fastapi import APIRouter, Depends
from sentence_transformers import SentenceTransformer

from rag_project.api.dependencies import get_embedding_model
from rag_project.db.session_manager import db_session_manager
from rag_project.db.session import SessionLocal
from rag_project.domain.enums import IngestionStatus
from rag_project.dto.models import IngestionRequest, IngestionResponse
from rag_project.tests.test_ingestion import create_ingestion_service

router = APIRouter()


class IngestionAPI:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    @db_session_manager
    def ingest_content(self, session, model, request: IngestionRequest) -> IngestionResponse:
        service = create_ingestion_service(session, model)

        chunks_count = service.ingest_content(
            source_type=request.source_type,
            source_path=request.source_path
        )

        return IngestionResponse(
            chunks_stored=chunks_count,
            status=IngestionStatus.SUCCESS
        )


ingestion_api = IngestionAPI(SessionLocal)


@router.post("/ingest", response_model=IngestionResponse)
def ingest_endpoint(
        request: IngestionRequest,
        model: SentenceTransformer = Depends(get_embedding_model)
) -> IngestionResponse:
    return ingestion_api.ingest_content(
        model=model,
        request=request
    )
