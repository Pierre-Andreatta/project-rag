import os
import tempfile
from urllib.parse import urlparse, parse_qs

import whisper
import torch
import yt_dlp
import time

from typing import Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from rag_project.exceptions import TranscriptionError
from rag_project.logger import get_logger


logger = get_logger(__name__)


class TranscriptionService:
    def __init__(self, model_size="base", languages: Optional[list[str]] = None):
        self.languages = languages or ["fr", "en"]
        self.model = whisper.load_model(model_size)

    @staticmethod
    def _extract_video_id(url: str) -> str:
        parsed = urlparse(url)
        if 'youtu.be' in parsed.netloc:
            return parsed.path.lstrip('/')
        if 'youtube.com' in parsed.netloc:
            query = parse_qs(parsed.query)
            return query.get('v', [None])[0]
        return url

    @staticmethod
    def _download_audio(video_url: str, output_path: str = "temp_audio.mp4"):
        logger.info(f"Downloading audio from {video_url}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path.replace('.mp3', ''),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '96'
            }],
            'extract_flat': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

    def _try_youtube_transcript(self, video_id: str) -> Optional[str]:
        try:
            logger.info(f"Trying transcript {video_id}")
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            transcript = transcript_list.find_transcript(self.languages).fetch()
            return " ".join([snippet.text for snippet in transcript])
        except (NoTranscriptFound, TranscriptsDisabled):
            return None

    def _transcribe_with_whisper(self, audio_path: str) -> str:
        logger.info(f"Transcribing {audio_path}")
        transcribe_params = {
            'task': 'transcribe',
            'fp16': torch.cuda.is_available(),  # If GPU
            'beam_size': 3,  # Quality reduction
        }
        result = self.model.transcribe(audio_path, **transcribe_params)
        return result["text"]

    def transcribe_youtube(self, video_url: str) -> str:

        video_id = self._extract_video_id(video_url)

        # Try YouTube Transcript API
        text = self._try_youtube_transcript(video_id)

        if text:
            logger.info(f"YouTube from transcription : {video_id}")
            return text

        # TODO: replace by redis when available
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Fallback Whisper
        try:
            self._download_audio(video_url=video_url, output_path=temp_path)
            start = time.time()
            text = self._transcribe_with_whisper(temp_path)
            logger.info(f"Whisper transcription took {time.time() - start:.2f}s")
        finally:
            if temp_file and os.path.exists(temp_path):
                os.remove(temp_path)

        if text:
            logger.info(f"YouTube from audio : {video_id}")
            return text
        else:
            raise TranscriptionError(f"Transcription for video {video_id} failed")
