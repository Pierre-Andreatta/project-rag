import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

from rag_project.logger import get_logger


logger = get_logger(__name__)

logger.info("START: session")

host = os.environ.get("POSTGRES_HOST", "localhost")
user = os.environ.get("POSTGRES_USER")
password = os.environ.get("POSTGRES_PASSWORD")

logger.info(f"Current host : {host}")

engine = create_engine(f"postgresql+psycopg://{user}:{password}@{host}:5432/db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
