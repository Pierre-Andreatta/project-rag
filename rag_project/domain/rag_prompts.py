from rag_project.domain.enums import LanguageEnum


class RagPrompt:
    def __init__(self, content: str, tokens: int):
        self.content = content
        self.tokens = tokens

    def format(self, question: str, context: str) -> str:
        return self.content.format(question=question, context=context)


class RagPromptFactory:
    _prompts = {
        LanguageEnum.FR: RagPrompt(
            content="Voici des documents :\n{context}\n\nQuestion : {question}\nRéponds de manière précise en t'appuyant uniquement sur ces documents.",
            tokens=25
        ),
        LanguageEnum.EN: RagPrompt(
            content="Here are some documents:\n{context}\n\nQuestion: {question}\nAnswer precisely, relying solely on these documents.",
            tokens=16
        )
    }

    @classmethod
    def get_prompt(cls, language: LanguageEnum) -> RagPrompt:
        return cls._prompts[language]
