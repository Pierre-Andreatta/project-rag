from fastapi import APIRouter, Depends

from rag_project.application.use_cases.IngestionUseCase import IngestionUseCase
from rag_project.domain.enums import IngestionStatus
from rag_project.domain.models.models import IngestionRequest, IngestionResponse
from rag_project.infrastructure.api.dependencies import get_ingestion_use_case

router = APIRouter()


@router.post("/ingest", response_model=IngestionResponse)
def ingest_endpoint(
        request: IngestionRequest,
        use_case: IngestionUseCase = Depends(get_ingestion_use_case),
) -> IngestionResponse:
    chunks_count = use_case.ingest_content(
        source_type=request.source_type,
        source_path=request.source_path,
    )
    return IngestionResponse(
        chunks_stored=chunks_count,
        status=IngestionStatus.SUCCESS,
    )
