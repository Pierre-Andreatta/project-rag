import requests
from bs4 import BeautifulSoup


def scrape_page(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html5lib")

    # Tags to remove
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    raw_texts = list(soup.stripped_strings)

    cleaned_texts = []
    seen = set()
    for text in raw_texts:
        text = " ".join(text.split())  # Normalize whitespace
        if len(text) < 15:  # Skip short content (likely noise)
            continue
        if text.lower() in seen:  # Avoid exact duplicates (case-insensitive)
            continue
        seen.add(text.lower())
        cleaned_texts.append(text)

    return "\n\n".join(cleaned_texts)
