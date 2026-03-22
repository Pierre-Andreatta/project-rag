from abc import ABC, abstractmethod


class TranscriptionInterface(ABC):

    @abstractmethod
    def transcribe_youtube(self, video_url: str) -> str:
        pass
