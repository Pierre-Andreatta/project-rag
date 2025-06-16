from typing import List

from sentence_transformers import SentenceTransformer

from rag_project.db.crud.content import ContentCRUD
from rag_project.db.session import SessionLocal
from rag_project.db.session_manager import db_session_manager
from rag_project.domain.models import SourceTypeEnum
from rag_project.exceptions import IngestionError
from rag_project.logger import get_logger
from rag_project.services.scraping_service import default_scraper
from rag_project.utils.text_processing import default_chunker


logger = get_logger(__name__)


def embed_chunks(model: SentenceTransformer, chunks: List) -> List:
    if not chunks:
        raise IngestionError(f"No chunks to embed")
    return model.encode(chunks, normalize_embeddings=True).tolist()


def ingest_chunks(session: SessionLocal, source_type: SourceTypeEnum, chunks: List, embeddings: List, source_path: str) -> int:
    try:
        if len(chunks) != len(embeddings):
            raise ValueError("nb chunks <> nb embeddings")

        content_crud = ContentCRUD(session)
        count = content_crud.store_chunks(chunks, embeddings, source_path, source_type)

        if count != len(chunks):
            raise ValueError("Unexpected chunks count")

        return count

    except Exception as e:
        logger.error(f"ingest_chunks : {str(e)}")
        raise


class IngestionService:
    def __init__(
            self,
            session_factory=SessionLocal,
            scraper=None,
            chunker=None
    ):
        self.session_factory = session_factory
        self.scraper = scraper or default_scraper
        self.chunker = chunker or default_chunker

    def content_from_youtube(self, youtube_url: str):
        raise NotImplementedError("YouTube ingestion not implemented yet")

    def content_from_pdf(self, path: str):
        raise NotImplementedError("Local ingestion not implemented yet")

    @db_session_manager
    def ingest_content(self,
                       session: SessionLocal, model: SentenceTransformer, source_type: SourceTypeEnum,
                       source_path: str) -> int:

        try:
            # Source handling
            if source_type is None:
                raise IngestionError("Source type must be provided", exc_info=False)

            if source_path is None:
                raise IngestionError("Source path must be provided", exc_info=False)

            text = None
            if source_type == SourceTypeEnum.WEB:
                text = self.scraper(url=source_path)
            elif source_type == SourceTypeEnum.YOUTUBE:
                text = self.content_from_youtube(source_path)
            elif source_type == SourceTypeEnum.PDF:
                text = self.content_from_pdf(source_path)

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
