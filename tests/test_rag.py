from typing import List

from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from rag_project.application.ports.embeddings.embedding_interface import EmbeddingInterface
from rag_project.domain.enums import LanguageEnum
from rag_project.domain.models.models import DocumentDto, SourceDto, AnswerDto
from rag_project.exceptions import RagError, ValidationError, DataBaseError
from rag_project.infrastructure.db.repositories.content_repository import ContentRepository
from rag_project.infrastructure.db.repositories.source_repository import SourceRepository
from rag_project.infrastructure.embedding.SentenceTransformerEmbeddingAdapter import SentenceTransformerEmbeddingAdapter
from rag_project.infrastructure.llm.openAILLMAdapter import OpenAILLMAdapter
from rag_project.logger import get_logger
from rag_project.domain.prompts.rag_prompts import RagPromptFactory
from rag_project.application.ports.llm.llm_interface import LLMInterface

logger = get_logger(__name__)


class RagService:
    def __init__(
            self,
            embedding_adapter: EmbeddingInterface,
            llm_adapter: LLMInterface,
            min_similarity: float = 0.4,
            token_limit: int = 4096
    ):
        self.embedding_adapter = embedding_adapter
        self.llm_adapter = llm_adapter
        self.min_similarity = min_similarity
        self.token_limit = token_limit

    @staticmethod
    def _validate_search_params(query_vector: List[float], top_k: int, min_k: int):
        """Valide les paramètres de recherche"""
        if not query_vector:
            raise ValidationError("Query vector cannot be empty", field='query_vector')

        if top_k <= 0:
            raise ValidationError("top_k must be positive", field='top_k')

        if min_k <= 0:
            raise ValidationError("min_k must be positive", field='min_k')

        if min_k > top_k:
            raise ValidationError("min_k cannot be greater than top_k", field='top_k, min_k')

    @staticmethod
    def _validate_question_params(question: str, top_k: int, min_k: int):
        """Valide les paramètres de la question"""
        if not question or not question.strip():
            raise ValidationError("Question cannot be empty", field='question')

        if len(question.strip()) < 5:
            raise ValidationError("Question is too short (minimum 5 characters)", field='question')

        if top_k <= 0 or min_k <= 0:
            raise ValidationError("top_k and min_k must be positive", field='top_k, min_k')

    @staticmethod
    def _enrich_with_sources(documents: List[DocumentDto], source_repository: SourceRepository) -> List[SourceDto]:
        """Enrichit les documents avec les informations des sources"""
        try:
            # Récupère les IDs uniques des sources
            source_ids = list(set(doc.id for doc in documents))

            # Récupère les sources depuis la DB
            sources = []
            for source_id in source_ids:
                source = source_repository.get_source_by_id(source_id)
                if source:
                    sources.append(source)
                else:
                    logger.warning(f"Source {source_id} not found")

            return sources

        except Exception as e:
            message = f"Failed to enrich documents with sources: {e}"
            logger.error(message)
            raise DataBaseError(message) from e

    # TODO: build prompt look complicate
    def build_optimized_prompt(
            self,
            question: str,
            documents: List[DocumentDto],
            language: LanguageEnum = LanguageEnum.FR
    ) -> str:
        # TODO: rework
        """Construit un prompt optimisé en tenant compte des limites de tokens"""
        try:
            # Utilise votre RagPromptFactory existant
            base_prompt = RagPromptFactory.get_default_prompt(question, documents, language)

            # Compte les tokens
            token_count = count_tokens(base_prompt)

            # Si le prompt dépasse la limite, on le tronque intelligemment
            if token_count > self.token_limit:
                logger.warning(f"Prompt too long ({token_count} tokens), truncating...")
                return self._truncate_prompt(question, documents, language)

            return base_prompt

        except Exception as e:
            message = f"Failed to build prompt: {e}"
            logger.error(message)
            raise RagError(message) from e

    def _truncate_prompt(
            self,
            question: str,
            documents: List[DocumentDto],
            language: LanguageEnum
    ) -> str:
        # TODO: rework
        """Tronque intelligemment le prompt pour respecter les limites de tokens"""
        # Réserve des tokens pour la question et les instructions
        reserved_tokens = 500
        available_tokens = self.token_limit - reserved_tokens

        # Tronque les documents jusqu'à respecter la limite
        truncated_docs = []
        current_tokens = 0

        for doc in documents:
            doc_tokens = count_tokens(doc.content)

            if current_tokens + doc_tokens <= available_tokens:
                truncated_docs.append(doc)
                current_tokens += doc_tokens
            else:
                # Tronque le document courant si possible
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 100:  # Minimum viable
                    truncated_content = self._truncate_text_to_tokens(
                        doc.content, remaining_tokens
                    )
                    truncated_doc = DocumentDto(
                        id=doc.id,
                        content=truncated_content,
                        source_id=doc.source_id,
                        embedding=doc.embedding
                    )
                    truncated_docs.append(truncated_doc)
                break

        return RagPromptFactory.get_default_prompt(question, truncated_docs, language)

    @staticmethod
    def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
        """Tronque un texte pour respecter une limite de tokens"""
        # Approximation simple : 1 token ≈ 0.75 mots en français
        max_chars = int(max_tokens * 4)  # Approximation

        if len(text) <= max_chars:
            return text

        # Tronque au niveau des phrases pour garder la cohérence
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')

        last_sentence_end = max(last_period, last_exclamation, last_question)

        if last_sentence_end > max_chars * 0.8:  # Si on peut garder une phrase complète
            return truncated[:last_sentence_end + 1]

        return truncated + "..."

    def search_similar_documents(
            self,
            session: Session,  # ✅ Session passée en paramètre
            query_vector: List[float],
            top_k: int,
            min_k: int
    ) -> tuple[List[DocumentDto], List[SourceDto]]:
        """Recherche des documents similaires"""
        try:
            self._validate_search_params(query_vector, top_k, min_k)

            content_repository = ContentRepository(session)
            source_repository = SourceRepository(session)

            documents = content_repository.find_similar_contents(
                query_vector, top_k, self.min_similarity
            )

            if len(documents) < min_k:
                raise RagError(
                    f'Not enough information to answer: {len(documents)} documents < {min_k}'
                )

            sources = self._enrich_with_sources(documents, source_repository)

            logger.info(f'Found {len(documents)} documents from {len(sources)} sources')
            return documents, sources

        except (ValidationError, DataBaseError):
            raise
        except Exception as e:
            message = f"Unexpected error in document search: {e}"
            logger.error(message)
            raise RagError(message) from e

    async def answer_question(
            self,
            session: Session,  # Session passée en paramètre
            question: str,
            top_k: int = 5,
            min_k: int = 1,
            language: LanguageEnum = LanguageEnum.FR
    ) -> AnswerDto:
        """Point d'entrée principal pour répondre à une question"""
        try:
            self._validate_question_params(question, top_k, min_k)

            logger.info(f"Processing question: {question[:100]}...")

            # Pipeline RAG
            query_vector = self.embedding_adapter.embed_single(question)
            documents, sources = self.search_similar_documents(
                session, query_vector, top_k, min_k
            )
            prompt = self.build_optimized_prompt(question, documents, language)
            answer = await self.llm_adapter.generate_response(prompt)

            return AnswerDto(answer=answer, sources=sources)

        except (ValidationError, DataBaseError, RagError):
            raise
        except Exception as e:
            message = f"Failed to answer question: {str(e)}"
            logger.error(message)
            raise RagError(message) from e


def create_rag_service(
        model: SentenceTransformer,
        llm_model: str = "gpt-3.5-turbo",
        min_similarity: float = 0.4,
        token_limit: int = 4096
) -> RagService:
    """Factory pour créer un service RAG configuré"""

    embedding_adapter = SentenceTransformerEmbeddingAdapter(model)
    llm_adapter = OpenAILLMAdapter(llm_model)

    return RagService(
        embedding_adapter=embedding_adapter,
        llm_adapter=llm_adapter,
        min_similarity=min_similarity,
        token_limit=token_limit
    )