from typing import List

from rag_project.adapters.content_extraction_adapter import ContentExtractorFactory
from rag_project.adapters.embedding_adapter import SentenceTransformerEmbeddingAdapter
from rag_project.adapters.storage_adapter import DatabaseStorageAdapter
from rag_project.db.session import SessionLocal
from rag_project.db.session_manager import db_session_manager
from rag_project.domain.enums import SourceTypeEnum
from rag_project.exceptions import IngestionError, ValidationError
from rag_project.logger import get_logger
from rag_project.services.scraping_service import default_scraper
from rag_project.services.transcription_service import TranscriptionService
from rag_project.utils.text_processing import default_chunker

logger = get_logger(__name__)


class IngestionService:
    def __init__(
            self,
            session_factory=SessionLocal,
            scraper=None,
            chunker=None,
            transcriber=None
    ):
        self.session_factory = session_factory
        self.scraper = scraper or default_scraper
        self.chunker = chunker or default_chunker
        self.transcriber = transcriber or TranscriptionService()

        # Factory pour les extractors de contenu
        self.content_extractor_factory = ContentExtractorFactory(
            scraper=self.scraper,
            transcriber=self.transcriber
        )

    def _extract_content(self, source_type: SourceTypeEnum, source_path: str) -> str:
        """Extract content using the appropriate adapter."""
        try:
            extractor = self.content_extractor_factory.get_extractor(source_type)
            return extractor.extract(source_path)

        except ValidationError:
            raise
        except IngestionError:
            raise
        except NotImplementedError:
            raise
        except Exception as e:
            message = f"Failed to extract content: {str(e)}"
            logger.error(message)
            raise IngestionError(message) from e

    def _chunk_text(self, text: str, max_tokens: int = 300) -> List[str]:
        """Chunk text into smaller pieces."""
        if not text:
            raise ValidationError("Text cannot be empty for chunking")

        try:
            chunks = self.chunker(text, max_tokens=max_tokens)
            if not chunks:
                raise IngestionError("No chunks created from text")

            logger.debug(f"Created {len(chunks)} chunks from text")
            return chunks

        except Exception as e:
            message = f"Failed to chunk text: {str(e)}"
            logger.error(message)
            raise IngestionError(message) from e

    def _embed_chunks(self, embedding_adapter, chunks: List[str]) -> List[List[float]]:
        """Embed chunks using the provided adapter."""
        try:
            logger.debug(f"Embedding {len(chunks)} chunks")
            embeddings = embedding_adapter.embed(chunks)
            logger.info(f"Successfully embedded {len(chunks)} chunks")
            return embeddings

        except Exception as e:
            message = f"Failed to embed chunks: {str(e)}"
            logger.error(message)
            raise IngestionError(message) from e

    def _store_chunks(self, storage_adapter, chunks: List[str], embeddings: List[List[float]],
                      source_path: str, source_type: SourceTypeEnum) -> int:
        """Store chunks using the provided adapter."""
        try:
            logger.debug(f"Storing {len(chunks)} chunks")
            count = storage_adapter.store_chunks(chunks, embeddings, source_path, source_type)
            logger.info(f"Successfully stored {count} chunks")
            return count

        except Exception as e:
            message = f"Failed to store chunks: {str(e)}"
            logger.error(message)
            raise IngestionError(message) from e

    @db_session_manager
    def ingest_content(self,
                       session: SessionLocal,
                       model,
                       source_type: SourceTypeEnum,
                       source_path: str,
                       max_tokens: int = 300) -> int:
        """
        Complete ingestion pipeline using all adapters.

        Args:
            session: Database session
            model: Embedding model (SentenceTransformer, etc.)
            source_type: Type of source (WEB, YOUTUBE, PDF)
            source_path: Path to the source
            max_tokens: Maximum tokens per chunk

        Returns:
            Number of chunks stored
        """
        try:
            # Validation des paramètres
            if source_type is None:
                raise ValidationError("Source type must be provided", field="source_type")
            if source_path is None:
                raise ValidationError("Source path must be provided", field="source_path")

            logger.info(f"Starting ingestion for {source_type} source: {source_path}")

            # Extract content via Content Extraction Adapter
            text = self._extract_content(source_type, source_path)

            # Chunk text
            chunks = self._chunk_text(text, max_tokens)

            # Embed chunks via Embedding Adapter
            embedding_adapter = SentenceTransformerEmbeddingAdapter(model)
            embeddings = self._embed_chunks(embedding_adapter, chunks)

            # Store chunks via Storage Adapter
            storage_adapter = DatabaseStorageAdapter(session)
            chunks_count = self._store_chunks(storage_adapter, chunks, embeddings, source_path, source_type)

            logger.info(f"Successfully ingested {chunks_count} chunks from {source_path}")
            return chunks_count

        except Exception as e:
            message = f"Ingestion failed for {source_path}: {str(e)}"
            logger.error(message)
            raise
