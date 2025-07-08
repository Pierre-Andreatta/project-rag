# TODO: set optimum max_tokens (see token_limite in rag_service)

from typing import List

from rag_project.exceptions import ValidationError, TextProcessingError
from rag_project.logger import get_logger

logger = get_logger(__name__)


def default_chunker(text: str, max_tokens=300) -> List[str]:

    try:

        if not text or not text.strip():
            raise ValidationError("text cannot be empty")

        if not isinstance(text, str):
            message = f"text must be string, not {type(text)}"
            logger.error(message)
            raise ValidationError(message)

        if max_tokens <= 0:
            message = "max_tokens must be positive"
            logger.error(message)
            raise ValidationError(message)

        sentences = text.split(". ")
        chunks = []
        current = ""
        for sent in sentences:
            if len(current.split()) + len(sent.split()) < max_tokens:
                current += sent + ". "
            else:
                chunks.append(current.strip())
                current = sent + ". "
        if current:
            chunks.append(current.strip())
        return chunks

    except ValidationError:
        raise
    except Exception as e:
        message = f"Failed to chunk text: {e}"
        logger.error(message)
        raise TextProcessingError(message) from e
