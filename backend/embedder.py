#backend/embedder.py

import os
import sqlite3
import uuid
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Paths
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "data"))
DB_PATH = os.path.join(DATA_DIR, "app.db")
INDEX_PATH = os.path.join(DATA_DIR, "indexes", "faiss.index")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

# Initialize embedding model (small, fast)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)

_index = None
_dim = _model.get_sentence_embedding_dimension()


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    with _get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            page_number INTEGER,
            chunk_type TEXT,
            content TEXT,
            metadata TEXT
        )
        """)
        conn.commit()


def _init_faiss():
    global _index
    if _index is not None:
        return _index
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
    else:
        _index = faiss.IndexFlatL2(_dim)
    return _index


def embed_texts(texts: List[str]) -> np.ndarray:
    return np.array(_model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True))


def add_chunks(chunks: List[Dict[str, Any]]):
    if not chunks:
        return

    init_db()
    index = _init_faiss()

    texts = [str(c["content"]) for c in chunks]  
    embeddings = embed_texts(texts)

    with _get_conn() as conn:
        for c, emb in zip(chunks, embeddings):
            chunk_id = str(c["chunk_id"]) if c.get("chunk_id") else uuid.uuid4().hex
            page_number = int(c.get("page_number", 0))  

            doc_id = str(c["doc_id"])
            chunk_type = str(c.get("chunk_type", "text"))
            content = str(c["content"])
            metadata = str(c.get("metadata", {}))

            conn.execute(
                """
                INSERT OR REPLACE INTO chunks(
                    chunk_id, doc_id, page_number, chunk_type, content, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, doc_id, page_number, chunk_type, content, metadata)
            )

            vec = np.expand_dims(emb, axis=0).astype("float32")
            index.add(vec)

        conn.commit()

    faiss.write_index(index, INDEX_PATH)


def query_index(query: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
    init_db()
    index = _init_faiss()

    q_emb = embed_texts([query]).astype("float32")
    D, I = index.search(q_emb, top_k)

    results = []

    with _get_conn() as conn:
        cur = conn.execute("SELECT chunk_id FROM chunks ORDER BY rowid")
        chunk_ids_in_order = [r[0] for r in cur.fetchall()]

        for faiss_idx, dist in zip(I[0], D[0]):
            if faiss_idx == -1 or faiss_idx >= len(chunk_ids_in_order):
                continue

            chunk_id = chunk_ids_in_order[faiss_idx]

            cur = conn.execute(
                "SELECT chunk_id, doc_id, page_number, chunk_type, content, metadata "
                "FROM chunks WHERE chunk_id = ?", 
                (chunk_id,)
            )
            row = cur.fetchone()
            if row:
                chunk_id, doc_id, page, ctype, content, metadata = row
                results.append((chunk_id, float(dist), {
                    "doc_id": doc_id,
                    "page_number": page,
                    "chunk_type": ctype,
                    "content": content,
                    "metadata": metadata
                }))

    return results



if __name__ == "__main__":
    init_db()
    sample_chunk = [{
        "chunk_id": uuid.uuid4().hex,
        "doc_id": "demo",
        "page_number": 1,
        "chunk_type": "text",
        "content": "Net profit for FY2024 was 12,345 INR",
        "metadata": {"source": "demo.pdf"}
    }]
    add_chunks(sample_chunk)
    res = query_index("What was the net profit in 2024?", top_k=2)
    print(res)