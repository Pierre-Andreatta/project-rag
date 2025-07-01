# TODO: set optimum token_limite

import os
from typing import List, Tuple
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt

from sentence_transformers import SentenceTransformer

from rag_project.db.crud.content import ContentCRUD
from rag_project.db.crud.source import SourceCRUD
from rag_project.db.session import SessionLocal
from rag_project.db.session_manager import db_session_manager
from rag_project.domain.enums import LanguageEnum
from rag_project.dto.models import DocumentDto, SourceDto, AnswerDto
from rag_project.exceptions import RagError
from rag_project.logger import get_logger

from rag_project.domain.rag_prompts import RagPromptFactory
from rag_project.utils.tokenizer import count_tokens

logger = get_logger(__name__)


def embed_question(model: SentenceTransformer, question: str) -> List:
    try:
        if len(question) < 5:
            raise RagError(f'Question {question} not valid')
        return model.encode(question, normalize_embeddings=True).tolist()
    except Exception:
        raise


class RagService:
    def __init__(
            self,
            session_factory=SessionLocal,
            llm_model="gpt-3.5-turbo",
            min_similarity: int = 0.4
    ):
        self.session_factory = session_factory
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_model = llm_model
        self.min_similarity = min_similarity

    def search_similar_documents(
            self,
            session: SessionLocal,
            query_vector: List,
            top_k: int,
            min_k: int
    ) -> tuple[List[DocumentDto], List[SourceDto]]:
        try:
            content_crud = ContentCRUD(session)
            source_crud = SourceCRUD(session)

            documents = content_crud.find_similar_contents(query_vector, top_k, self.min_similarity)

            if len(documents) < min_k:
                raise RagError('Not enough information to answer')

            sources: List[SourceDto] = []
            for document in documents:
                if document.source_data.id:
                    source = source_crud.get_source_by_id(document.source_data.id)
                    if source:
                        document.source_data = source
                        if source not in sources:
                            sources.append(source)

            logger.info(f'Found {len(documents)} documents')
            logger.info(f'Documents {[doc.__dict__ for doc in documents]}')
            return documents, sources
        except Exception as e:
            logger.error(f"search_similar_documents: {e}", exc_info=True)
            raise

    def trim_documents_to_fit_token_limit(self, docs: List[DocumentDto], base_tokens_count: int, token_limit: int):

        current_docs = docs.copy()
        context = "\n\n".join([doc.content for doc in current_docs])
        current_context_token_count = count_tokens(context, model_name=self.llm_model)

        while len(current_docs) > 1 and (base_tokens_count + current_context_token_count) > token_limit:
            last_doc_token_count = count_tokens(current_docs[-1].content, model_name=self.llm_model)
            current_docs = current_docs[:-1]
            current_context_token_count = current_context_token_count - last_doc_token_count

        token_count = base_tokens_count + current_context_token_count

        if token_count > token_limit:
            raise RagError(
                f"Prompt too long for LLM context - tokens in prompt: {token_count} > tokens limit: {token_limit}")

        logger.info(
            f"{len(docs) - len(current_docs)} documents removed to fit tokens limit: {token_limit} "
            f"(final prompt size: {token_count} tokens)"
        )

        context = "\n\n".join([doc.content for doc in current_docs])
        return current_docs, context

    def build_prompt(self, question: str, docs: List[DocumentDto], language: LanguageEnum = LanguageEnum.FR,
                     token_limit: int = 1600) -> str:
        try:

            if not isinstance(language, LanguageEnum):
                raise RagError(f'Language {language} not supported')

            prompt_obj = RagPromptFactory.get_prompt(language)

            base_tokens_count = prompt_obj.tokens + count_tokens(question, model_name=self.llm_model)

            current_docs, context = self.trim_documents_to_fit_token_limit(
                docs=docs, base_tokens_count=base_tokens_count, token_limit=token_limit
            )

            return prompt_obj.content.format(question=question, context=context)

        except Exception:
            raise

    @retry(stop=stop_after_attempt(3), reraise=True)
    async def query_llm_async(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise

    @db_session_manager
    async def answer_question(self, session: SessionLocal, model: SentenceTransformer, question: str,
                              top_k: int = 5, min_k: int = 1) -> AnswerDto:
        try:
            query_vector = embed_question(model, question)
            documents, sources = self.search_similar_documents(session, query_vector=query_vector, top_k=top_k,
                                                               min_k=min_k)
            prompt = self.build_prompt(question, docs=documents, language=LanguageEnum.FR)
            answer = await self.query_llm_async(prompt)

            return AnswerDto(answer=answer, sources=sources)

        except Exception as e:
            logger.error(f"answer_question : {str(e)}")
            raise
