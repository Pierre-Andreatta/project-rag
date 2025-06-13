from typing import List

from openai import OpenAI
import os

from rag_project.db.crud.content import ContentCRUD
from rag_project.db.session import get_session, SessionLocal
from rag_project.domain.models import DocumentDomain
from rag_project.logger import get_logger

logger = get_logger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# TODO: set prompt in other file to choice injected prompt
def build_prompt(question: str, docs: list[DocumentDomain]) -> str:
    try:
        context = "\n\n".join([doc.content for doc in docs])
        return f"Voici des documents :\n{context}\n\nQuestion : {question}\nRéponds de manière précise en t'appuyant uniquement sur ces documents."
    except Exception as e:
        logger.error(f"build_prompt : {e}")


def query_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"query_llm : {e}")


def search_similar_documents(query_vector, top_k=2) -> List[DocumentDomain]:
    session = SessionLocal()
    content_crud = ContentCRUD(session)
    documents_data = content_crud.find_similar_contents(query_vector, top_k)
    session.close()
    return [DocumentDomain(**doc_data) for doc_data in documents_data]
