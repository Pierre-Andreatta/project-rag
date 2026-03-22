from rag_project.application.ports.llm.llm_interface import LLMInterface


class AnthropicLLMAdapter(LLMInterface):
    async def generate_response(self, prompt: str, **kwargs) -> str:
        # Implementation avec Anthropic Claude
        raise NotImplementedError("Cette méthode n'est pas encore implémentée")

    @property
    def model(self) -> str:
        raise NotImplementedError("Cette méthode n'est pas encore implémentée")
