from typing import List

from sentence_transformers import SentenceTransformer

from rag_project.db.crud.content import ContentCRUD
from rag_project.db.session import SessionLocal
from rag_project.db.session_manager import db_session_manager
from rag_project.dto.models import SourceTypeEnum
from rag_project.exceptions import IngestionError, ValidationError, DataBaseError, TranscriptionError, ScraperError
from rag_project.logger import get_logger
from rag_project.services.scraping_service import default_scraper
from rag_project.services.transcription_service import TranscriptionService
from rag_project.utils.text_processing import default_chunker


logger = get_logger(__name__)


def embed_chunks(model: SentenceTransformer, chunks: List) -> List:

    if not chunks:
        raise ValidationError("No chunks provided for embedding", field="chunks")

    if not isinstance(chunks, list):
        raise ValidationError("Chunks must be a list", field="chunks")

    try:
        logger.debug(f"Embedding {len(chunks)} chunks")
        embeddings = model.encode(chunks, normalize_embeddings=True).tolist()

        if len(embeddings) != len(chunks):
            raise IngestionError(f"Embedding count mismatch: {len(embeddings)} != {len(chunks)}")

        logger.info(f"Successfully embedded {len(chunks)} chunks")
        return embeddings

    except Exception as e:
        message = f"Failed to embed chunks: {str(e)}"
        logger.error(message)
        raise IngestionError(message) from e


def ingest_chunks(session: SessionLocal, source_type: SourceTypeEnum, chunks: List, embeddings: List, source_path: str) -> int:

    if not chunks:
        raise ValidationError("No chunks provided for ingestion", field="chunks")

    if not embeddings:
        raise ValidationError("No embeddings provided for ingestion", field="embeddings")

    if len(chunks) != len(embeddings):
        raise ValidationError(f"Chunks count ({len(chunks)}) doesn't match embeddings count ({len(embeddings)})")

    if not source_path or not source_path.strip():
        raise ValidationError("Source path cannot be empty", field="source_path")

    if not isinstance(source_type, SourceTypeEnum):
        raise ValidationError(f"Invalid source type: {source_type}", field="source_type")

    try:
        logger.debug(f"Ingesting {len(chunks)} chunks from {source_path}")

        content_crud = ContentCRUD(session)
        count = content_crud.store_chunks(chunks, embeddings, source_path, source_type)

        if count != len(chunks):
            raise IngestionError("Unexpected stored chunks count")

        logger.info(f"Successfully ingested {count} chunks from {source_path}")
        return count

    except ValidationError:
        raise
    except DataBaseError:
        raise
    except Exception as e:
        message = f"Failed to ingest chunks {source_path}: {e}"
        logger.error(message)
        raise IngestionError(message) from e


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

    def content_from_youtube(self, youtube_url: str):

        if not youtube_url or not youtube_url.strip():
            raise ValidationError("YouTube URL cannot be empty", field="youtube_url")

        try:
            logger.debug(f"Extracting content from YouTube: {youtube_url}")
            content = self.transcriber.transcribe_youtube(youtube_url.strip())

            if not content or not content.strip():
                raise IngestionError(f"No content extracted from YouTube URL: {youtube_url}")

            logger.info(f"Successfully extracted {len(content)} characters from YouTube")
            return content.strip()

        except TranscriptionError as e:
            logger.error(f"Transcription error for YouTube URL {youtube_url}: {e}")
            raise IngestionError(f"Failed to transcribe YouTube video: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error extracting YouTube content: {e}", exc_info=True)
            raise IngestionError(f"Failed to extract YouTube content: {str(e)}") from e

    def content_from_web(self, url: str) -> str:

        if not url or not url.strip():
            raise ValidationError("URL cannot be empty", field="url")

        try:
            logger.info(f"Scraping content from web: {url}")
            content = self.scraper(url=url.strip())

            if not content or not content.strip():
                raise IngestionError(f"No content extracted from URL: {url}")

            logger.info(f"Successfully scraped {len(content)} characters from web")
            return content.strip()

        except ScraperError as e:
            logger.error(f"Scraping error for URL {url}: {e}")
            raise IngestionError(f"Failed to scrape web content: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error scraping web content: {e}", exc_info=True)
            raise IngestionError(f"Failed to extract web content: {str(e)}") from e

    def content_from_pdf(self, path: str):
        raise NotImplementedError("Local ingestion not implemented yet")

    def _extract_content_by_type(self, source_type: SourceTypeEnum, source_path: str) -> str:

        try:
            if source_type == SourceTypeEnum.WEB:
                return self.content_from_web(source_path)
            elif source_type == SourceTypeEnum.YOUTUBE:
                return self.content_from_youtube(source_path)
            elif source_type == SourceTypeEnum.PDF:
                return self.content_from_pdf(source_path)
            else:
                raise ValidationError(f"Unsupported source type: {source_type}", field="source_type")

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

    @db_session_manager
    def ingest_content(self,
                       session: SessionLocal, model: SentenceTransformer, source_type: SourceTypeEnum,
                       source_path: str) -> int:

        try:
            # Source handling
            if source_type is None:
                raise ValidationError("Source type must be provided", field="source_type")

            if source_path is None:
                raise ValidationError("Source path must be provided", field="source_path")

            text = self._extract_content_by_type(source_type, source_path)

            if text is None:
                raise IngestionError(f"No texts from source {source_type}", exc_info=False)

            chunks = self.chunker(text, max_tokens=300)
            embeddings = embed_chunks(model, chunks)
            chunks_count = ingest_chunks(session, source_type, chunks, embeddings, source_path)

            return chunks_count

        except Exception as e:
            message = f"ingest_content : {str(e)}"
            logger.error(message)
            raise
