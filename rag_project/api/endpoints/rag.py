from fastapi import APIRouter, Depends
from sentence_transformers import SentenceTransformer

from rag_project.api.dependencies import get_embedding_model
from rag_project.db.session_manager import db_session_manager
from rag_project.db.session import SessionLocal
from rag_project.domain.enums import LanguageEnum
from rag_project.dto.models import ChatRequest, ChatResponse
from rag_project.tests.test_rag import create_rag_service

router = APIRouter()


class RagAPI:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    @db_session_manager
    async def chat(self, session, model, request: ChatRequest) -> ChatResponse:
        """Chat endpoint avec gestion de session"""
        service = create_rag_service(
            model=model,
            llm_model=request.llm_model or "gpt-3.5-turbo",
            min_similarity=request.min_similarity or 0.4
        )

        answer = await service.answer_question(
            session=session,
            question=request.question,
            top_k=request.top_k or 5,
            min_k=request.min_k or 1,
            language=request.language or LanguageEnum.FR
        )

        return ChatResponse(
            response=answer.answer,
            sources=[{
                "id": source.id,
                "path": source.path,
                "type": source.type
            } for source in answer.sources]
        )


rag_api = RagAPI(SessionLocal)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    model: SentenceTransformer = Depends(get_embedding_model)
) -> ChatResponse:
    return await rag_api.chat(
        model=model,
        request=request
    )
