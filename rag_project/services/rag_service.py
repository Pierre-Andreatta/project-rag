# TODO: set optimum token_limite

import os
from typing import List, Tuple
from openai import AsyncOpenAI, OpenAIError
from tenacity import retry, stop_after_attempt

from sentence_transformers import SentenceTransformer

from rag_project.db.crud.content import ContentCRUD
from rag_project.db.crud.source import SourceCRUD
from rag_project.db.session import SessionLocal
from rag_project.db.session_manager import db_session_manager
from rag_project.domain.enums import LanguageEnum
from rag_project.dto.models import DocumentDto, SourceDto, AnswerDto
from rag_project.exceptions import RagError, ValidationError, EmbeddingError, DataBaseError
from rag_project.logger import get_logger

from rag_project.domain.rag_prompts import RagPromptFactory
from rag_project.utils.tokenizer import count_tokens

logger = get_logger(__name__)


def embed_question(model: SentenceTransformer, question: str) -> List:
    # TODO: move to adapters
    try:
        if not question or not question.strip():
            raise ValidationError("Question cannot be empty")

        if len(question.strip()) < 5:
            raise ValidationError(f"Question '{question}' is too short (minimum 5 characters)")

        return model.encode(question, normalize_embeddings=True).tolist()

    except ValidationError:
        raise
    except Exception as e:
        message = f"embed_question: {e}"
        logger.error(message)
        raise EmbeddingError(message) from e


class RagService:
    def __init__(
            self,
            session_factory=SessionLocal,
            llm_model="gpt-3.5-turbo",
            min_similarity: int = 0.4
    ):
        self.session_factory = session_factory
        self.llm_model = llm_model
        self.min_similarity = min_similarity

        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValidationError("OPENAI_API_KEY environment variable not set")
            self.client = AsyncOpenAI(api_key=api_key)
        except Exception as e:
            message = f"Failed to initialize OpenAI client: {e}"
            logger.error(message)
            raise ValidationError(message) from e

    def search_similar_documents(
            self,
            session: SessionLocal,
            query_vector: List,
            top_k: int,
            min_k: int
    ) -> tuple[List[DocumentDto], List[SourceDto]]:
        try:

            if not query_vector:
                raise ValidationError("Query vector cannot be empty")

            if top_k <= 0 or min_k <= 0:
                raise ValidationError("top_k and min_k must be positive integers")

            if min_k > top_k:
                raise ValidationError("min_k cannot be greater than top_k")

            content_crud = ContentCRUD(session)
            source_crud = SourceCRUD(session)

            documents = content_crud.find_similar_contents(query_vector, top_k, self.min_similarity)

            if len(documents) < min_k:
                raise RagError(f'Not enough information to answer: {len(documents)} documents < {min_k}')

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

        except (ValidationError, DataBaseError):
            raise
        except Exception as e:
            message = f"Unexpected error in document search: {e}"
            logger.error(message)
            raise RagError(message) from e

    def trim_documents_to_fit_token_limit(self, docs: List[DocumentDto], base_tokens_count: int, token_limit: int):

        try:

            if not docs:
                raise ValidationError("Documents list cannot be empty")

            if base_tokens_count < 0 or token_limit <= 0:
                raise ValidationError("Invalid token counts")

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

        except ValidationError:
            raise
        except Exception as e:
            message = f"Failed to trim documents: {str(e)}"
            logger.error(message)
            raise RagError(message) from e

    def build_prompt(self, question: str, docs: List[DocumentDto], language: LanguageEnum = LanguageEnum.FR,
                     token_limit: int = 4096) -> str:
        try:
            # Validation des paramètres
            if not question or not question.strip():
                raise ValidationError("Question cannot be empty")

            if not docs:
                raise ValidationError("Documents list cannot be empty")

            if not isinstance(language, LanguageEnum):
                raise ValidationError(f"Unsupported language: {language}")

            if token_limit <= 0:
                raise ValidationError("Token limit must be positive")

            prompt_obj = RagPromptFactory.get_prompt(language)

            base_tokens_count = prompt_obj.tokens + count_tokens(question, model_name=self.llm_model)

            current_docs, context = self.trim_documents_to_fit_token_limit(
                docs=docs, base_tokens_count=base_tokens_count, token_limit=token_limit
            )

            return prompt_obj.content.format(question=question, context=context)

        except ValidationError:
            raise
        except Exception as e:
            message = f"Failed to build prompt: {str(e)}"
            logger.error(message)
            raise RagError(message) from e

    @retry(stop=stop_after_attempt(3), reraise=True)
    async def query_llm_async(self, prompt: str) -> str:
        # TODO: move to adapters/llm_adapter.py
        try:

            if not prompt or not prompt.strip():
                raise ValidationError("Prompt cannot be empty")

            logger.debug(f"Querying LLM with model: {self.llm_model}")

            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )

            if not response.choices or not response.choices[0].message.content:
                # raise LLMError("Empty response from LLM")
                raise RagError("Empty response from LLM")

            return response.choices[0].message.content

        except ValidationError:
            raise
        except OpenAIError as e:
            message = f"OpenAI API error: {e}"
            logger.error(message)
            # raise LLMError(message) from e
            raise RagError(message) from e
        except RagError as e:
            message = f"Failed to query LLM: {e}"
            logger.error(message)
            # raise LLMError(message) from e
            raise RagError(message) from e

    @db_session_manager
    async def answer_question(self, session: SessionLocal, model: SentenceTransformer, question: str,
                              top_k: int = 5, min_k: int = 1) -> AnswerDto:
        try:

            if not question or not question.strip():
                raise ValidationError("Question cannot be empty")

            if not model:
                raise ValidationError("Model cannot be None")

            if top_k <= 0 or min_k <= 0:
                raise ValidationError("top_k and min_k must be positive")

            logger.info(f"Processing question: {question[:100]}...")

            query_vector = embed_question(model, question)
            documents, sources = self.search_similar_documents(session, query_vector=query_vector, top_k=top_k,
                                                               min_k=min_k)
            prompt = self.build_prompt(question, docs=documents, language=LanguageEnum.FR)
            answer = await self.query_llm_async(prompt)

            return AnswerDto(answer=answer, sources=sources)

        except (ValidationError, EmbeddingError, DataBaseError):  # add SearchError, PromptError, LLMError
            raise
        except Exception as e:
            message = f"Failed to answer question: {str(e)}"
            logger.error(message)
            raise RagError(message) from e
