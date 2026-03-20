# TODO: add filed to ValidationError

from fastapi import FastAPI, Query, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

from rag_project.api.dependencies import get_embedding_model, get_ingestion_service, get_rag_service
from rag_project.api.exception_handlers import ingestion_exception_handler, database_exception_handler, \
    unexpected_exception_handler, validation_exception_handler
from rag_project.exceptions import IngestionError, DataBaseError, TimeOutError, RagError, ValidationError, \
    UnexpectedError
from rag_project.logger import get_logger
from rag_project.services.ingestion_service import IngestionService
from rag_project.services.rag_service import RagService


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    # Initialize SentenceTransformer
    fast_api_app.state.model = SentenceTransformer("all-MiniLM-L6-v2")  # type: ignore
    yield

app = FastAPI(
    lifespan=lifespan,
    title="rag-project",
    version="1.0.0",
    description="Retrieval-augmented generation API project"
)

app.add_exception_handler(IngestionError, ingestion_exception_handler)
app.add_exception_handler(DataBaseError, database_exception_handler)
app.add_exception_handler(UnexpectedError, unexpected_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)


@app.post("/ask")
async def ask_question(
        question: str = Query(...),
        model: SentenceTransformer = Depends(get_embedding_model),
        service: RagService = Depends(get_rag_service)
):
    try:

        if not question.strip():
            raise ValidationError("Question cannot be empty")

        answer = await service.answer_question(
            model=model,
            question=question
        )
        return {"answer": answer.answer, "sources": [
            {
                'source_type': source.source_type,
                'source_path': source.source_path
            } for source in answer.sources
        ]}
        # TODO: uncomment when front deployed
        # return {"answer": answer.answer, "sources": answer.sources}

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RagError as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except DataBaseError as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    except TimeOutError as e:
        logger.error(f"Timeout error: {e}")
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Unexpected error during question processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
