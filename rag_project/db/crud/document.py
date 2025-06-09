import json

from sqlalchemy import cast
from sqlalchemy.sql import literal

from rag_project.db.session import get_session
from rag_project.db.models.document_models import DocumentORM, Vector


def get_similar_documents(query_vector, top_k):
    with get_session() as session:
        embedding_cast = cast(literal(query_vector), Vector(384))
        distance_op = DocumentORM.embedding.op("<->")(embedding_cast)

        results = (
            session.query(DocumentORM)
            .order_by(distance_op)
            .limit(top_k)
            .all()
        )
        return [
            {
                "id": row.id,
                "content": row.content,
                "embedding": row.embedding,
                "meta": row.meta
            }
            for row in results
        ]


def store_chunks(chunks, embeddings, source_url):
    with get_session() as session:
        docs = [
            DocumentORM(
                content=text,
                meta={"source": source_url},
                embedding=emb
            )
            for text, emb in zip(chunks, embeddings)
        ]
        session.bulk_save_objects(docs)
        session.commit()
