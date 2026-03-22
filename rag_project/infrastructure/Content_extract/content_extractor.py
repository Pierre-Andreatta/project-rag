from abc import abstractmethod

from rag_project.application.ports.content_extraction.content_extraction_interface import ContentExtractorInterface, ContentExtractorFactoryInterface
from rag_project.application.ports.transcription.transcriptionInterface import TranscriptionInterface
from rag_project.domain.enums import SourceTypeEnum
from rag_project.exceptions import TranscriptionError, IngestionError, ValidationError, ScraperError
from rag_project.logger import get_logger


logger = get_logger(__name__)


class ContentExtractor(ContentExtractorInterface):
    @abstractmethod
    def extract(self, source_path: str) -> str:
        pass

    @staticmethod
    def _validate_source(source_path: str, field_name: str = "source_path") -> str:
        """Common validation logic for all extractors."""
        if not source_path or not source_path.strip():
            raise ValidationError(f"{field_name} cannot be empty", field=field_name)
        return source_path.strip()

    @staticmethod
    def _validate_content(content: str, source_path: str) -> str:
        """Common content validation logic."""
        if not content or not content.strip():
            raise IngestionError(f"No content extracted from: {source_path}")
        return content.strip()


class WebContentExtractorInterface(ContentExtractor):
    def __init__(self, scraper):
        self.scraper = scraper

    def extract(self, url: str) -> str:
        url = self._validate_source(url, "url")

        try:
            logger.info(f"Starting web content extraction from: {url}")
            content = self.scraper(url=url)
            content = self._validate_content(content, source_path=url)
            logger.info(f"Successfully extracted {len(content)} characters from web")
            return content
        except ScraperError as e:
            message = f"Failed to scrape web content: {str(e)}"
            logger.error(message)
            raise IngestionError(message) from e


class YoutubeContentExtractorInterface(ContentExtractor):
    def __init__(self, transcriber: TranscriptionInterface):
        self.transcriber = transcriber

    def extract(self, youtube_url: str) -> str:
        youtube_url = self._validate_source(youtube_url, "youtube_url")

        try:
            logger.info(f"Starting YouTube transcription from: {youtube_url}")
            content = self.transcriber.transcribe_youtube(youtube_url)
            content = self._validate_content(content, source_path=youtube_url)
            logger.info(f"Successfully transcribed {len(content)} characters from YouTube")
            return content
        except TranscriptionError as e:
            message = f"Failed to transcribe YouTube video: {str(e)}"
            logger.error(f"YouTube transcription failed for {youtube_url}: {message}")
            raise IngestionError(message) from e


class PDFContentExtractorInterface(ContentExtractor):
    def extract(self, path: str):
        path = self._validate_source(path, "file_path")
        raise NotImplementedError("PDF extraction not implemented yet")


class ContentExtractorFactory(ContentExtractorFactoryInterface):
    def __init__(self, scraper, transcriber):
        self.extractors = {
            SourceTypeEnum.WEB: WebContentExtractorInterface(scraper),
            SourceTypeEnum.YOUTUBE: YoutubeContentExtractorInterface(transcriber),
            SourceTypeEnum.PDF: PDFContentExtractorInterface()
        }

    def get_extractor(self, source_type: SourceTypeEnum) -> ContentExtractor:
        if source_type not in self.extractors:
            raise ValidationError(f"Unsupported source type: {source_type}")
        return self.extractors[source_type]
