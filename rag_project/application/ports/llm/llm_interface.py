from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        pass
