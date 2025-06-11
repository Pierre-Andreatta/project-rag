from sentence_transformers import SentenceTransformer

from rag_project.db.crud.content import ContentCRUD
from rag_project.db.session import SessionLocal
from rag_project.exceptions import IngestionError
from rag_project.services.scraping_service import default_scraper
from rag_project.utils.text_processing import default_chunker


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
        self.source_url = url
        self.texts = self.scraper(url)

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

    def ingest_chunks(self, category_id: int = 1):
        with self.session_factory() as session:
            try:
                if len(self.chunks) != len(self.embeddings):
                    raise ValueError("nb chunks <> nb embeddings")

                crud = ContentCRUD(session)
                count = crud.store_chunks(self.chunks, self.embeddings, self.source_url, category_id)

                if count != len(self.chunks):
                    raise ValueError("Unexpected chunks count")

                return count
            except Exception as e:
                session.rollback()
                self.reset_state()
                raise IngestionError(f"ingest_chunks : {str(e)}") from e

    def ingest_content(self,
                       model: SentenceTransformer, category_id: int = None,
                       url: str = None, youtube_url: str = None, path: str = None):

        self.reset_state()

        # Source handling
        if sum(x is not None for x in [url, youtube_url, path]) != 1:
            raise IngestionError("Exactly one source must be provided")

        if url:
            self.content_from_url(url)
        elif youtube_url:
            self.content_from_youtube(youtube_url)
        elif path:
            self.content_from_local(path)

        self.chunk_text(max_tokens=300)
        self.embed_chunks(model)
        return self.ingest_chunks(category_id)
