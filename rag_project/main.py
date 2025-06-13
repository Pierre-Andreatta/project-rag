from fastapi import FastAPI, Query, Request, Depends, HTTPException
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

from rag_project.api.dependencies import get_embedding_model, get_ingestion_service
from rag_project.domain.models import SourceTypeEnum
from rag_project.exceptions import IngestionError, DataBaseError, TimeOutError
from rag_project.services.ingestion_service import IngestionService
from rag_project.services.rag import build_prompt, query_llm, search_similar_documents


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


@app.post("/ask")
def ask_question(request: Request, question: str = Query(...)):
    model = request.app.state.model
    query_vector = model.encode(question, normalize_embeddings=True).tolist()
    docs = search_similar_documents(query_vector, top_k=5)
    prompt = build_prompt(question, docs)
    answer = query_llm(prompt)
    return {"answer": answer}
