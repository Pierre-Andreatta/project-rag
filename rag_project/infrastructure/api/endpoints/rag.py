from fastapi import APIRouter, Depends, HTTPException, Query
from sentence_transformers import SentenceTransformer

from rag_project.application.use_cases.RagUseCase import RagUseCase
from rag_project.exceptions import DataBaseError, RagError, TimeOutError, ValidationError
from rag_project.infrastructure.api.dependencies import get_embedding_model, get_rag_use_case
from rag_project.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["rag"])


@router.post("/ask")
async def ask_question(
        question: str = Query(...),
        model: SentenceTransformer = Depends(get_embedding_model),
        service: RagUseCase = Depends(get_rag_use_case),
):
    try:
        if not question.strip():
            raise ValidationError("Question cannot be empty")

        answer = await service.answer_question(
            model=model,
            question=question,
        )
        return {"answer": answer.answer, "sources": [
            {
                "source_type": source.source_type,
                "source_path": source.source_path,
            }
            for source in answer.sources
        ]}

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
