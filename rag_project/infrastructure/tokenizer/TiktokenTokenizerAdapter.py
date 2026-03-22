# TODO: move to infrastructure layer

import tiktoken
from functools import lru_cache

from rag_project.application.ports.tokenizer.TokenizerInterface import TokenizerInterface
from rag_project.logger import get_logger

logger = get_logger(__name__)


class TiktokenTokenizerAdapter(TokenizerInterface):

    def __init__(self, model_name: str = None):
        self.model_name = model_name or "gpt-3.5-turbo"

    @lru_cache(maxsize=4)
    def _get_tokenizer(self) -> tiktoken.core.Encoding:
        # Works only with OpenAI
        logger.info(f"Initializing tokenizer for model: {self.model_name}")
        return tiktoken.encoding_for_model(self.model_name)

    def count_tokens(self, text: str) -> int:
        try:
            tokenizer = self._get_tokenizer()
            return len(tokenizer.encode(text))
        except Exception as e:
            message = f"count_tokens: {e}"
            logger.error(message)
            raise message
