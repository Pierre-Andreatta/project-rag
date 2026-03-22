import os
from openai import AsyncOpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_project.exceptions import ValidationError, LLMError
from rag_project.main import logger
from rag_project.application.ports.llm.llm_interface import LLMInterface


class OpenAILLMAdapter(LLMInterface):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self._model = model

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValidationError("OPENAI_API_KEY environment variable not set")

        self.client = AsyncOpenAI(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            if not prompt or not prompt.strip():
                raise ValidationError("Prompt cannot be empty")

            logger.debug(f"Querying LLM with model: {self.model}")

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )

            if not response.choices or not response.choices[0].message.content:
                raise LLMError("Empty response from LLM")

            return response.choices[0].message.content

        except ValidationError:
            raise
        except OpenAIError as e:
            message = f"OpenAI API error: {e}"
            logger.error(message)
            raise LLMError(message) from e
        except Exception as e:
            message = f"Failed to query LLM: {e}"
            logger.error(message)
            raise LLMError(message) from e