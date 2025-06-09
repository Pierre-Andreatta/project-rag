from fastapi import FastAPI, Query, Request
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

from rag_project.db.crud.document import store_chunks
from rag_project.services.chunk import chunk_text, embed_chunks
from rag_project.services.scraping import scrape_page
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
def ingest_url(request: Request, url: str = Query(...)):
    model = request.app.state.model
    text = scrape_page(url)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks, model)
    store_chunks(chunks, embeddings, url)
    return {"message": f"{len(chunks)} chunks ingested from {url}"}


@app.post("/ask")
def ask_question(request: Request, question: str = Query(...)):
    model = request.app.state.model
    query_vector = model.encode(question, normalize_embeddings=True).tolist()
    docs = search_similar_documents(query_vector, top_k=5)
    prompt = build_prompt(question, docs)
    answer = query_llm(prompt)
    return {"answer": answer}
