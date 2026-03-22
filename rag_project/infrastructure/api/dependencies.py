from functools import lru_cache

from fastapi import Depends
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from rag_project.application.ports.content_extraction.content_extraction_interface import ContentExtractorFactoryInterface
from rag_project.application.use_cases.IngestionUseCase import IngestionUseCase
from rag_project.application.use_cases.RagUseCase import RagUseCase
from rag_project.infrastructure.db.repositories.content_repository import ContentRepository
from rag_project.infrastructure.db.repositories.source_repository import SourceRepository
from rag_project.infrastructure.db.session import get_session
from rag_project.infrastructure.embedding.SentenceTransformerEmbeddingAdapter import SentenceTransformerEmbeddingAdapter
from rag_project.infrastructure.llm.openAILLMAdapter import OpenAILLMAdapter
from rag_project.infrastructure.tokenizer.TiktokenTokenizerAdapter import TiktokenTokenizerAdapter


@lru_cache(maxsize=1)
def _cached_content_extractor_factory() -> ContentExtractorFactoryInterface:
    from rag_project.infrastructure.Content_extract.content_extractor import ContentExtractorFactory
    from rag_project.infrastructure.scraping.scraper import default_scraper
    from rag_project.infrastructure.transcription.WhisperTranscriptionAdapter import WhisperTranscriptionAdapter

    return ContentExtractorFactory(default_scraper, WhisperTranscriptionAdapter())


def get_content_extractor_factory() -> ContentExtractorFactoryInterface:
    return _cached_content_extractor_factory()


def get_ingestion_use_case(
        session: Session = Depends(get_session),
        content_extractor_factory: ContentExtractorFactoryInterface = Depends(get_content_extractor_factory),
) -> IngestionUseCase:
    return IngestionUseCase(
        embedder=SentenceTransformerEmbeddingAdapter(),
        content_repository=ContentRepository(session),
        content_extractor_factory=content_extractor_factory,
    )


def get_rag_use_case(session: Session = Depends(get_session)) -> RagUseCase:
    return RagUseCase(
        content_repository=ContentRepository(session),
        source_repository=SourceRepository(session),
        embedder=SentenceTransformerEmbeddingAdapter(),
        llm=OpenAILLMAdapter(),
        tokenizer=TiktokenTokenizerAdapter()
    )
