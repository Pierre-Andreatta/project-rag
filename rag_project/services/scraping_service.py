import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from rag_project.exceptions import ScraperError


def default_scraper(url: str, timeout: int = 4) -> str:
    try:
        if not urlparse(url).scheme in ('http', 'https'):
            raise ScraperError("Invalid URL scheme")

        response = requests.get(url, timeout=timeout)
        soup = BeautifulSoup(response.content, "html5lib")

        # Tags to remove
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()

        raw_texts = list(soup.stripped_strings)

        texts = []
        seen = set()
        for text in raw_texts:
            text = " ".join(text.split())  # Normalize whitespace
            if len(text) < 15:  # Skip short content (likely noise)
                continue
            if text.lower() in seen:  # Avoid exact duplicates (case-insensitive)
                continue
            seen.add(text.lower())
            texts.append(text)

        return "\n\n".join(texts)

    except Exception:
        raise
