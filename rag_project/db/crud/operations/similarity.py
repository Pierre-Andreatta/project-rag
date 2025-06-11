from typing import List
from sqlalchemy import cast, literal
from sqlalchemy.orm import Session
from rag_project.db.models.content import ContentORM, Vector


# TODO: maybe move to crud.content.py
def find_similar_contents(
        session: Session,
        query_vector: List[float],
        top_k: int = 5,
        min_similarity: float = 0.7
) -> List[dict]:

    embedding_cast = cast(literal(query_vector), Vector(384))
    distance_op = ContentORM.embedding.cosine_distance(embedding_cast)

    results = (
        session.query(ContentORM)
        .filter(distance_op < (1 - min_similarity))
        .order_by(distance_op)
        .limit(top_k)
        .all()
    )

    return [{
        'id': r.id,
        'content': r.content,
        'similarity': 1 - distance_op,
        'source': r.source.path_to_content if r.source else None
    } for r in results]
