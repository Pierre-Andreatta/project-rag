# TODO: add filed to ValidationError

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from sentence_transformers import SentenceTransformer

from rag_project.infrastructure.api.endpoints.ingestion import router as ingestion_router
from rag_project.infrastructure.api.endpoints.rag import router as rag_router
from rag_project.infrastructure.api.exception_handlers import ingestion_exception_handler, database_exception_handler, \
    unexpected_exception_handler, validation_exception_handler
from rag_project.exceptions import IngestionError, DataBaseError, UnexpectedError


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

app.include_router(ingestion_router)
app.include_router(rag_router)
