from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from rag_project.application.ports.content_extraction.content_extraction_interface import ContentExtractorFactoryInterface
from rag_project.application.use_cases.ingestion_use_case import IngestionUseCase
from rag_project.application.use_cases.rag_use_case import RagUseCase
from rag_project.infrastructure.db.repositories.postgres.content_repository import PostgresContentRepository
from rag_project.infrastructure.db.repositories.postgres.source_repository import PostgresSourceRepository
from rag_project.infrastructure.db.session import get_session
from rag_project.infrastructure.embedding.sentence_transformer_embedding_adapter import SentenceTransformerEmbeddingAdapter
from rag_project.infrastructure.llm.open_ai_llm_adapter import OpenAILLMAdapter
from rag_project.infrastructure.tokenizer.tiktoken_tokenizer_adapter import TiktokenTokenizerAdapter


@lru_cache(maxsize=1)
def _cached_content_extractor_factory() -> ContentExtractorFactoryInterface:
    from rag_project.infrastructure.content_extract.content_extractor import ContentExtractorFactory
    from rag_project.infrastructure.scraping.scraper import default_scraper
    from rag_project.infrastructure.transcription.whisper_transcription_adapter import WhisperTranscriptionAdapter

    return ContentExtractorFactory(default_scraper, WhisperTranscriptionAdapter())


def get_content_extractor_factory() -> ContentExtractorFactoryInterface:
    return _cached_content_extractor_factory()


def get_ingestion_use_case(
        session: Session = Depends(get_session),
        content_extractor_factory: ContentExtractorFactoryInterface = Depends(get_content_extractor_factory),
) -> IngestionUseCase:
    return IngestionUseCase(
        embedder=SentenceTransformerEmbeddingAdapter(),
        content_repository=PostgresContentRepository(session),
        content_extractor_factory=content_extractor_factory,
    )


def get_rag_use_case(session: Session = Depends(get_session)) -> RagUseCase:
    return RagUseCase(
        content_repository=PostgresContentRepository(session),
        source_repository=PostgresSourceRepository(session),
        embedder=SentenceTransformerEmbeddingAdapter(),
        llm=OpenAILLMAdapter(),
        tokenizer=TiktokenTokenizerAdapter()
    )
