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

from rag_project.utils.rag_prompts import rag_prompt_fr, rag_prompt_en

logger = get_logger(__name__)


class RagService:
    def __init__(
            self,
            session_factory=SessionLocal,
            llm_model="gpt-4-turbo"
    ):
        self.session_factory = session_factory
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_model = llm_model
        self.min_confidence = 0.7
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
            documents_data = content_crud.find_similar_contents(self.query_vector, top_k)
            if len(documents_data) < min_k:
                raise RagError('Not enough information to answer')
            self.docs = [DocumentDomain(**doc_data) for doc_data in documents_data]
            logger.info(f'docs {self.docs}')
        except Exception:
            raise

    def build_prompt(self, question: str, language: LanguageEnum = LanguageEnum.FR, token_limite: int = 1600):
        try:
            context = "\n\n".join([doc.content for doc in self.docs])
            if language == LanguageEnum.EN:
                self.prompt = rag_prompt_en
            elif language == LanguageEnum.FR:
                self.prompt = rag_prompt_fr
            else:
                raise RagError(f'Language {language} not supported')

            self.prompt = self.prompt.format(question=question, context=context)

            if len(self.prompt) > token_limite:
                raise RagError("Prompt too long for LLM context")

        except Exception:
            raise

    # def query_llm(self):
    #     try:
    #         response = self.client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[{"role": "user", "content": self.prompt}]
    #         )
    #         self.answer = response.choices[0].message.content
    #     except Exception as e:
    #         raise

    @retry(stop=stop_after_attempt(3))
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
    def answer_question(self, session: SessionLocal, model: SentenceTransformer, question: str,
                        top_k: int = 6, min_k: int = 1) -> str:
        try:
            self.reset_state()

            self.embed_question(model, question)
            self.search_similar_documents(session, top_k=top_k, min_k=min_k)
            self.query_llm_async()

            return self.answer

        except Exception as e:
            message = f"answer_question : {str(e)}"
            logger.error(message)
            raise
