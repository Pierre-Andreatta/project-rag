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
        self.texts = ''
        self.chunks = []
        self.embeddings = []
        self.source_url = ''

    def reset_state(self):
        # Reset processes states
        self.texts = ''
        self.chunks = []
        self.embeddings = []
        self.source_url = ''

    def content_from_url(self, url: str):
        try:
            self.source_url = url
            self.texts = self.scraper(url)

        except Exception:
            raise

    def content_from_youtube(self, youtube_url: str):
        raise NotImplementedError("YouTube ingestion not implemented yet")

    def content_from_local(self, path: str):
        raise NotImplementedError("Local ingestion not implemented yet")

    def chunk_text(self, max_tokens: int):
        if not self.texts:
            raise IngestionError(f"No texts to chunk")
        self.chunks = self.chunker(self.texts, max_tokens)

    def embed_chunks(self, model: SentenceTransformer):
        if not self.chunks:
            raise IngestionError(f"No chunks to embed")
        self.embeddings = model.encode(self.chunks, normalize_embeddings=True).tolist()

    def ingest_chunks(self, session: SessionLocal, source_type: SourceTypeEnum) -> int:
        try:
            if len(self.chunks) != len(self.embeddings):
                raise ValueError("nb chunks <> nb embeddings")

            content_crud = ContentCRUD(session)
            count = content_crud.store_chunks(self.chunks, self.embeddings, self.source_url, source_type)

            if count != len(self.chunks):
                raise ValueError("Unexpected chunks count")

            return count

        except Exception as e:
            self.reset_state()
            message = f"ingest_chunks : {str(e)}"
            logger.error(message)
            raise

    @db_session_manager
    def ingest_content(self,
                       session: SessionLocal, model: SentenceTransformer, source_type: SourceTypeEnum,
                       url: str = None, youtube_url: str = None, path: str = None) -> int:

        try:
            self.reset_state()

            # Source handling
            if source_type is None:
                raise IngestionError("Source type must be provided", exc_info=False)

            if sum(x is not None for x in [url, youtube_url, path]) != 1:
                raise IngestionError("Exactly one source must be provided", exc_info=False)

            if url:
                self.content_from_url(url)
            elif youtube_url:
                self.content_from_youtube(youtube_url)
            elif path:
                self.content_from_local(path)

            self.chunk_text(max_tokens=300)
            self.embed_chunks(model)
            chunks_count = self.ingest_chunks(session, source_type)

            return chunks_count

        except Exception as e:
            message = f"ingest_content : {str(e)}"
            logger.error(message)
            raise
