from sentence_transformers import SentenceTransformer


def chunk_text(text: str, max_tokens: int = 300):
    sentences = text.split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current.split()) + len(sentence.split()) < max_tokens:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks


def embed_chunks(chunks, model: SentenceTransformer):
    return [model.encode(chunk, normalize_embeddings=True).tolist() for chunk in chunks]
