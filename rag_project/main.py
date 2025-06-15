from fastapi import FastAPI, Query, Depends, HTTPException
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

from rag_project.api.dependencies import get_embedding_model, get_ingestion_service, get_rag_service
from rag_project.domain.models import SourceTypeEnum
from rag_project.exceptions import IngestionError, DataBaseError, TimeOutError, RagError
from rag_project.services.ingestion_service import IngestionService
from rag_project.services.rag_service import RagService


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


@app.post("/ingest-url")
async def ingest_url(
        url: str,
        model: SentenceTransformer = Depends(get_embedding_model),
        service: IngestionService = Depends(get_ingestion_service)
):
    try:
        count = service.ingest_content(
            model=model,
            url=url,
            source_type=SourceTypeEnum.WEB
        )
        return {"status": "success", "ingested": count}

    except IngestionError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except DataBaseError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except TimeOutError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(
        question: str = Query(...),
        model: SentenceTransformer = Depends(get_embedding_model),
        service: RagService = Depends(get_rag_service)
):
    try:
        answer = await service.answer_question(
            model=model,
            question=question
        )
        return {"answer": answer}

    except RagError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except DataBaseError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except TimeOutError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
