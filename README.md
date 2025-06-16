# contentToRag

A Retrieval-Augmented Generation (RAG) project using PostgreSQL with `pgvector`, FastAPI, OpenAI API, and SQLAlchemy.

---

## 🚀 Features

- Vector storage of documents using `pgvector`
  - Only from html pages for the moment
- `FastAPI`-based API to query vectorized content
- Integration with `OpenAI`'s GPT models (GPT-4o / GPT-4o-mini)
- Database migrations with `Alembic`
- `Dockerized` environment
- Built-in logging system

## ⚙️ Requirements

- Docker & Docker Compose
- An OpenAI API key

---

## 🔧 Configuration

Create a `.env` file at the root of the project:

```env
OPENAI_API_KEY=sk-...
POSTGRES_DB=db
POSTGRES_USER=user...
POSTGRES_PASSWORD=secret...
```

---

## ▶️ Running the Project

```
docker-compose up -d --build
```

---

## 🌐 API Endpoints
Interactive documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Endpoints:
- ```/ingest-url``` : used to fetch the content of a web page by providing its URL.  
The content is split into chunks, which are then stored in the database with embeddings for semantic search.


- ```/ask```: used to query the application. It retrieves the top-N chunks with the closest embeddings.
These retrieved chunks are then sent to an OpenAI API, which uses only them to generate a summarized answer.

---

## 📂 Database Migrations
Create a new migration manually:
```
alembic revision --autogenerate -m "Message..."
```
Apply migrations manually:
```
alembic upgrade head
```
### 🔄 Automatic migration with Docker:
When using Docker, Alembic automatically runs the migrations during container startup.
This is handled by the command section in the docker-compose.yml, which includes:
```
alembic upgrade head
```
So no manual migration step is needed when using docker-compose up.

---

## ✉️ Logging
A logger is configured in rag_project/logger.py

---

## 🖥️ Tech Stack
- `FastAPI`
- `SQLAlchemy`
- `Alembic`
- `OpenAI API`
- `pgvector`
- `postgres`
- `Docker`

---

## 🎯️ TODO

- [ ] **Summarizing texts:**  
Consider the value of summarizing texts before embedding and storing them in the database.  
This would allow for more concise chunks.  
The loss of information would be offset by the ability to include more in the prompt.


- [ ] **Clean text before embedding:**
```python
unicodedata.normalize(...)
re.sub(...)
```

- [ ] **Manage duplicated or similar content:**

  - [ ] Detect similar contents in app or db with embedding
```python
from scipy.spatial.distance import cosine
import numpy as np

SIMILARITY_THRESHOLD = 0.95

def is_too_similar(new_emb, existing_embs):
    return any(1 - cosine(new_emb, emb) > SIMILARITY_THRESHOLD for emb in existing_embs)
```
or
```sql
select *
from table
WHERE embedding <-> :emb < 0.05
```


 - [ ] **Prefilter duplicated content with text** ***(only if duplicated from different sources.*** **To monitor)**  

Create hash column
```sql
class ContentORM(Base):
    __tablename__ = "contents"
    ...
    hash = Column(String(64), unique=True, index=True)  # SHA-256 = 64 hex
    ...
```
```filter``` before ```insert```
```python
import hashlib

def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```
```python
for text in chunks:
    text_hash = compute_text_hash(text)
    ...
    if session.query(ContentORM).filter_by(hash=text_hash).first():
        continue  # Skip
    ...
    content = ContentORM(content=text, hash=text_hash, embedding=emb)
    ...
```

- [X] **Split document table** 

```ContentORM```
```python
class ContentORM(Base):
    __tablename__ = "contents"

    id = Column(Integer, primary_key=True)
    content = Column(String, unique=True)
    embedding = Column(Vector(384))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

```SourceORM```
```python
class SourceORM(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True)
    url = Column(String, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

```SourceContentORM``` association table
```python
class SourceContentORM(Base):
    __tablename__ = "source_contents"

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("sources.id"), index=True)
    content_id = Column(Integer, ForeignKey("contents.id"), index=True)
    is_accepted = Column(Boolean, default=True, index=True)
    reason = Column(String, nullable=True)  # e.g. : "too_similar"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("ix_source_content_unique", "source_id", "content_id", unique=True),
    )

    source = relationship("SourceORM", backref="source_contents")
    content = relationship("ContentORM", backref="source_links")
```

- [x] **Vectorial index on embedding** (e.g. with ivfflat, hnsw)

```sql
CREATE INDEX ON contents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```
```lists = n``` Adapte ```n``` to project
- [ ] **Add support for other document types:**
  - [ ] PDF (e.g. with PyMuPDF, pdfplumber)
  - [ ] DOCX (e.g. with docx2txt, python-docx)
  - [ ] Youtube (e.g. with yt_dlp + Whisper, yt-dlp + youtube_transcript_api)
  - [ ] TXT (e.g. with chardet)
  - [ ] ...


- [ ] **Implement async and multiprocessing**
  - [x] async
  - [ ] multiprocessing


- [ ] **Create test suite with pytest**


- [ ] **Add a whitelist/registry of already processed URLs to avoid re-processing the same content.**


- [ ] **Add CI/CD with GitHub Actions**


- [ ] **Optimize with NumPy (vectors, np.dot, np.linalg.norm, np.array)**


- [ ] **Implement frontend interface**