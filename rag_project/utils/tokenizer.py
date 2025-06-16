import tiktoken
from functools import lru_cache

from rag_project.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str) -> tiktoken.core.Encoding:
    # Works only with OpenAI
    logger.info(f"Initializing tokenizer for model: {model_name}")
    return tiktoken.encoding_for_model(model_name)


def count_tokens(text: str, model_name: str) -> int:
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text))
