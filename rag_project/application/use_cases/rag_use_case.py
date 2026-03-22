from typing import List

from rag_project.application.ports.embeddings.embedding_interface import EmbeddingInterface
from rag_project.application.ports.llm.llm_interface import LLMInterface
from rag_project.application.ports.repositories.content_repository_interface import ContentRepositoryInterface
from rag_project.application.ports.repositories.source_repository_interface import SourceRepositoryInterface
from rag_project.application.ports.tokenizer.tokenizer_interface import TokenizerInterface
from rag_project.domain.enums import LanguageEnum
from rag_project.domain.models.models import DocumentDto, SourceDto, AnswerDto
from rag_project.exceptions import RagError, ValidationError, EmbeddingError, DataBaseError, LLMError
from rag_project.logger import get_logger

from rag_project.domain.prompts.rag_prompts import RagPromptFactory

logger = get_logger(__name__)


class RagUseCase:
    def __init__(
            self,
            content_repository: ContentRepositoryInterface,
            source_repository: SourceRepositoryInterface,
            embedder: EmbeddingInterface,
            llm: LLMInterface,
            tokenizer: TokenizerInterface,
            min_similarity: int = 0.4,
    ):
        self.content_repository = content_repository
        self.source_repository = source_repository
        self.embedder = embedder
        self.llm = llm
        self.tokenizer = tokenizer
        self.min_similarity = min_similarity

    def search_similar_documents(
            self,
            query_vector: List,
            top_k: int,
            min_k: int,
    ) -> tuple[List[DocumentDto], List[SourceDto]]:
        try:

            if not query_vector:
                raise ValidationError("Query vector cannot be empty")

            if top_k <= 0 or min_k <= 0:
                raise ValidationError("top_k and min_k must be positive integers")

            if min_k > top_k:
                raise ValidationError("min_k cannot be greater than top_k")

            documents = self.content_repository.find_similar_contents(
                query_vector, top_k, self.min_similarity
            )

            if len(documents) < min_k:
                raise RagError(f'Not enough information to answer: {len(documents)} documents < {min_k}')

            sources: List[SourceDto] = []
            for document in documents:
                if document.source_data.id:
                    source = self.source_repository.get_source_by_id(document.source_data.id)
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
            current_context_token_count = self.tokenizer.count_tokens(context)

            while len(current_docs) > 1 and (base_tokens_count + current_context_token_count) > token_limit:
                last_doc_token_count = self.tokenizer.count_tokens(current_docs[-1].content)
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

            base_tokens_count = prompt_obj.tokens + self.tokenizer.count_tokens(question)

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

    async def answer_question(
            self,
            question: str,
            top_k: int = 5,
            min_k: int = 1,
    ) -> AnswerDto:
        try:

            if not question or not question.strip():
                raise ValidationError("Question cannot be empty")

            if top_k <= 0 or min_k <= 0:
                raise ValidationError("top_k and min_k must be positive")

            logger.info(f"Processing question: {question[:100]}...")

            query_vector = self.embedder.embed([question])
            documents, sources = self.search_similar_documents(
                query_vector=query_vector, top_k=top_k, min_k=min_k
            )
            prompt = self.build_prompt(question, docs=documents, language=LanguageEnum.FR)
            answer = await self.llm.generate_response(prompt)

            return AnswerDto(answer=answer, sources=sources)

        except (ValidationError, EmbeddingError, DataBaseError, LLMError):  # add SearchError, PromptError
            raise
        except Exception as e:
            message = f"Failed to answer question: {str(e)}"
            logger.error(message)
            raise RagError(message) from e
