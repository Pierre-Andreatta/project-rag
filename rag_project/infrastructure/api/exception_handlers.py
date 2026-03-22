from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from rag_project.exceptions import IngestionError, DataBaseError, UnexpectedError
from rag_project.logger import get_logger

logger = get_logger(__name__)

# TODO: add rag_exception_handler


async def ingestion_exception_handler(request: Request, exc: IngestionError):
    logger.error(f"Ingestion error: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={
            "detail": exc.message,
            "error_code": exc.code,
            "type": "ingestion_error"
        }
    )


async def database_exception_handler(request: Request, exc: DataBaseError):
    logger.error(f"Database error: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Database error occurred",
            "error_code": exc.code,
            "type": "database_error"
        }
    )


async def unexpected_exception_handler(request: Request, exc: UnexpectedError):
    logger.error(f"Unexpected error: {exc.message}")
    return JSONResponse(
        status_code=exc.code,
        content={
            "detail": "Internal server error",
            "error_code": exc.code,
            "type": "unexpected_error"
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
            "type": "validation_error"
        }
    )