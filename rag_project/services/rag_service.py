# TODO: set optimum token_limite

import os
from typing import List
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt

from sentence_transformers import SentenceTransformer

from rag_project.db.crud.content import ContentCRUD
from rag_project.db.session import SessionLocal
from rag_project.db.session_manager import db_session_manager
from rag_project.domain.models import DocumentDomain, LanguageEnum
from rag_project.exceptions import RagError
from rag_project.logger import get_logger

from rag_project.domain.rag_prompts import RagPromptFactory
from rag_project.utils.tokenizer import count_tokens

logger = get_logger(__name__)


class RagService:
    def __init__(
            self,
            session_factory=SessionLocal,
            llm_model="gpt-3.5-turbo"
    ):
        self.session_factory = session_factory
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_model = llm_model
        self.min_similarity = 0.3
        self.query_vector = []
        self.docs = []
        self.prompt = []
        self.answer = ''

    def reset_state(self):
        # Reset processes states
        self.query_vector = []
        self.docs: List[DocumentDomain] = []
        self.prompt = ''
        self.answer = ''

    def embed_question(self, model: SentenceTransformer, question: str):
        try:
            if len(question) < 5:
                raise RagError(f'Question {question} not valid')
            self.query_vector = model.encode(question, normalize_embeddings=True).tolist()
        except Exception:
            raise

    def search_similar_documents(self, session: SessionLocal, top_k: int, min_k: int):
        try:
            content_crud = ContentCRUD(session)
            documents_data = content_crud.find_similar_contents(self.query_vector, top_k, self.min_similarity)
            if len(documents_data) < min_k:
                raise RagError('Not enough information to answer')
            self.docs = [DocumentDomain(**doc_data) for doc_data in documents_data]
            logger.info(f'Found {len(self.docs)} documents')
        except Exception:
            raise

    def adapt_prompt_to_fit_token_limit(self, docs: List[DocumentDomain], base_tokens_count, token_limit):

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

    def build_prompt(self, question: str, language: LanguageEnum = LanguageEnum.FR, token_limit: int = 1600):
        try:

            if not isinstance(language, LanguageEnum):
                raise RagError(f'Language {language} not supported')

            prompt_obj = RagPromptFactory.get_prompt(language)

            base_tokens_count = prompt_obj.tokens + count_tokens(question, model_name=self.llm_model)

            current_docs, context = self.adapt_prompt_to_fit_token_limit(
                docs=self.docs, base_tokens_count=base_tokens_count, token_limit=token_limit
            )

            self.prompt = prompt_obj.content.format(question=question, context=context)

        except Exception:
            raise

    @retry(stop=stop_after_attempt(3), reraise=True)
    async def query_llm_async(self):
        try:
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": self.prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise

    @db_session_manager
    async def answer_question(self, session: SessionLocal, model: SentenceTransformer, question: str,
                              top_k: int = 5, min_k: int = 1) -> str:
        try:
            self.reset_state()

            self.embed_question(model, question)
            self.search_similar_documents(session, top_k=top_k, min_k=min_k)
            self.build_prompt(question, language=LanguageEnum.FR)
            self.answer = await self.query_llm_async()

            return self.answer

        except Exception as e:
            message = f"answer_question : {str(e)}"
            logger.error(message)
            raise
