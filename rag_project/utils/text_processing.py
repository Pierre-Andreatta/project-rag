# TODO: set optimum max_tokens (see token_limite in rag_service)

from typing import List


def default_chunker(text: str, max_tokens=300) -> List[str]:
    """Implémentation plus robuste du chunking"""
    sentences = text.split(". ")
    chunks = []
    current = ""
    for sent in sentences:
        if len(current.split()) + len(sent.split()) < max_tokens:
            current += sent + ". "
        else:
            chunks.append(current.strip())
            current = sent + ". "
    if current:
        chunks.append(current.strip())
    return chunks
