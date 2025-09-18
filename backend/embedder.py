# backend/embedder.py

# import os
# import sqlite3
# import uuid
# from typing import List, Dict, Any, Tuple

# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# import torch
# import json
# from db import utils as db_utils  # ✅ needed for rebuild_index

# # ----------------------------
# # Paths
# # ----------------------------
# DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "data"))
# DB_PATH = os.path.join(DATA_DIR, "app.db")
# INDEX_PATH = os.path.join(DATA_DIR, "indexes", "faiss.index")

# os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
# os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

# # ----------------------------
# # Model + Index
# # ----------------------------
# # MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# device= "cuda" if torch.cuda.is_available() else "cpu"
# _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# _index = None
# _dim = _model.get_sentence_embedding_dimension()


# def _get_conn():
#     conn = sqlite3.connect(DB_PATH)
#     conn.execute("PRAGMA journal_mode=WAL;")
#     return conn


# def init_db():
#     with _get_conn() as conn:
#         conn.execute("""
#         CREATE TABLE IF NOT EXISTS chunks (
#             chunk_id TEXT PRIMARY KEY,
#             doc_id TEXT,
#             page_number INTEGER,
#             chunk_type TEXT,
#             content TEXT,
#             metadata TEXT
#         )
#         """)
#         conn.commit()


# def _init_faiss():
#     """Initialize or load FAISS index"""
#     global _index
#     if _index is not None:
#         return _index
#     if os.path.exists(INDEX_PATH):
#         _index = faiss.read_index(INDEX_PATH)
#     else:
#         _index = faiss.IndexFlatL2(_dim)
#     return _index


# def clear_index():
#     """Clear FAISS index and delete saved file"""
#     global _index
#     _index = faiss.IndexFlatL2(_dim)
#     if os.path.exists(INDEX_PATH):
#         os.remove(INDEX_PATH)
#     print("🗑️ FAISS index cleared.")


# def embed_texts(texts: List[str]) -> np.ndarray:
#     """Encode a batch of texts into embeddings"""
#     return np.array(
#         _model.encode(
#             texts,
#             batch_size=32,
#             show_progress_bar=False,
#             normalize_embeddings=True
#         )
#     )


# def add_chunks(chunks: List[Dict[str, Any]]):
#     """Add document chunks into SQLite + FAISS index"""
#     if not chunks:
#         return

#     init_db()
#     index = _init_faiss()

#     texts = [str(c["content"]) for c in chunks]
#     embeddings = embed_texts(texts)

#     with _get_conn() as conn:
#         for c, emb in zip(chunks, embeddings):
#             chunk_id = str(c.get("chunk_id") or uuid.uuid4().hex)
#             page_number = int(c.get("page_number", 0))

#             doc_id = str(c.get("doc_id", ""))
#             chunk_type = str(c.get("chunk_type", "text"))
#             content = str(c["content"])
#             metadata = str(c.get("metadata", {}))

#             # Save chunk in DB
#             conn.execute(
#                 """
#                 INSERT OR REPLACE INTO chunks(
#                     chunk_id, doc_id, page_number, chunk_type, content, metadata
#                 ) VALUES (?, ?, ?, ?, ?, ?)
#                 """,
#                 (chunk_id, doc_id, page_number, chunk_type, content, metadata)
#             )

#             # Add vector to FAISS
#             vec = np.expand_dims(emb, axis=0).astype("float32")
#             index.add(vec)

#         conn.commit()

#     # Persist index
#     faiss.write_index(index, INDEX_PATH)


# def query_index(query: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
#     """Query FAISS index and return top matching chunks"""
#     init_db()
#     index = _init_faiss()

#     q_emb = embed_texts([query]).astype("float32")
#     D, I = index.search(q_emb, top_k)

#     results = []

#     with _get_conn() as conn:
#         cur = conn.execute("SELECT chunk_id FROM chunks ORDER BY rowid")
#         chunk_ids_in_order = [r[0] for r in cur.fetchall()]

#         for faiss_idx, dist in zip(I[0], D[0]):
#             if faiss_idx == -1 or faiss_idx >= len(chunk_ids_in_order):
#                 continue

#             chunk_id = chunk_ids_in_order[faiss_idx]

#             cur = conn.execute(
#                 "SELECT chunk_id, doc_id, page_number, chunk_type, content, metadata "
#                 "FROM chunks WHERE chunk_id = ?",
#                 (chunk_id,)
#             )
#             row = cur.fetchone()
#             if row:
#                 chunk_id, doc_id, page, ctype, content, metadata = row
#                 try:
#                     metadata = json.loads(metadata) if metadata else {}
#                 except Exception:
#                     metadata = {}
#                 results.append((chunk_id, float(dist), {
#                     "doc_id": doc_id,
#                     "page_number": page,
#                     "chunk_type": ctype,
#                     "content": content,
#                     "metadata": metadata
#                 }))

#     return results


# def rebuild_index(batch_size: int = 64):
#     """
#     Rebuild the FAISS ANN index from scratch using all chunks in the database.
#     """
#     print("⚡ Rebuilding ANN index from scratch...")

#     clear_index()

#     # Fetch all chunks from DB
#     chunks = db_utils.get_all_chunks()
#     print(f"📦 Found {len(chunks)} chunks in DB to re-index")

#     if not chunks:
#         print("⚠️ No chunks found in DB. Did you run pop_chunks.py?")
#         return

#     # Re-embed and add to index
#     texts = [c["content"] for c in chunks]
#     ids = [c["chunk_id"] for c in chunks]

#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i+batch_size]
#         batch_chunks = [chunks[j] for j in range(i, i+len(batch_texts))]
#         embeddings = embed_texts(batch_texts)
#         add_chunks(batch_chunks)

#     print("✅ ANN index rebuilt successfully.")


# if __name__ == "__main__":
#     init_db()
#     sample_chunk = [{
#         "chunk_id": uuid.uuid4().hex,
#         "doc_id": "demo",
#         "page_number": 1,
#         "chunk_type": "text",
#         "content": "Net profit for FY2024 was 12,345 INR",
#         "metadata": {"source": "demo.pdf"}
#     }]
#     add_chunks(sample_chunk)
#     res = query_index("What was the net profit in 2024?", top_k=2)
#     print(res)

import os
import sqlite3
import uuid
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from db import utils as db_utils

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "data"))
DB_PATH = os.path.join(DATA_DIR, "app.db")
INDEX_PATH = os.path.join(DATA_DIR, "indexes", "faiss.index")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Embedder device:", device)
_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

_index = None
_dim = _model.get_sentence_embedding_dimension()

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    db_utils.init_db()  # ensures chunks + faiss_map etc exist

def _init_faiss():
    global _index
    if _index is not None:
        return _index
    if os.path.exists(INDEX_PATH):
        try:
            _index = faiss.read_index(INDEX_PATH)
            return _index
        except Exception as e:
            print("Failed to read faiss index:", e)
    # create ID map wrapper around flat L2
    base = faiss.IndexFlatL2(_dim)
    _index = faiss.IndexIDMap(base)
    return _index

def clear_index():
    global _index
    _index = faiss.IndexIDMap(faiss.IndexFlatL2(_dim))
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    with _get_conn() as conn:
        conn.execute("DELETE FROM faiss_map")
        conn.commit()
    print("🗑️ FAISS index and mapping cleared.")

def embed_texts(texts: List[str]) -> np.ndarray:
    # returns float32 vectors (not normalized necessarily, but we normalize to unit vectors)
    arr = _model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.array(arr).astype("float32")

def _get_next_faiss_id(conn) -> int:
    cur = conn.execute("SELECT MAX(faiss_id) FROM faiss_map")
    row = cur.fetchone()
    if row and row[0] is not None:
        return int(row[0]) + 1
    return 1

def add_chunks(chunks: List[Dict[str, Any]]):
    """
    Adds chunks to DB and FAISS index using explicit faiss_ids stored in faiss_map.
    """
    if not chunks:
        return
    init_db()
    index = _init_faiss()

    texts = [str(c.get("content", "")) for c in chunks]
    embeddings = embed_texts(texts)

    with _get_conn() as conn:
        to_add_vecs = []
        to_add_ids = []
        # prepare DB inserts first and build mapping
        for c, emb in zip(chunks, embeddings):
            chunk_id = str(c.get("chunk_id") or uuid.uuid4().hex)
            page_number = int(c.get("page_number", 0) or 0)
            doc_id = str(c.get("doc_id") or "")
            chunk_type = str(c.get("chunk_type") or "text")
            content = str(c.get("content") or "")
            metadata_obj = c.get("metadata", {}) or {}
            # ensure dict -> JSON string
            try:
                metadata_json = json.dumps(metadata_obj, ensure_ascii=False)
            except Exception:
                metadata_json = json.dumps(str(metadata_obj))

            # Save/replace chunk in DB
            conn.execute("""
                INSERT OR REPLACE INTO chunks(chunk_id, doc_id, page_number, chunk_type, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk_id, doc_id, page_number, chunk_type, content, metadata_json))

            # Ensure faiss_map has an id for this chunk
            cur = conn.execute("SELECT faiss_id FROM faiss_map WHERE chunk_id = ?", (chunk_id,))
            r = cur.fetchone()
            if r and r[0] is not None:
                faiss_id = int(r[0])
            else:
                faiss_id = _get_next_faiss_id(conn)
                conn.execute("INSERT OR REPLACE INTO faiss_map(faiss_id, chunk_id) VALUES (?, ?)", (faiss_id, chunk_id))

            to_add_vecs.append(emb.astype("float32"))
            to_add_ids.append(faiss_id)

        conn.commit()

    # convert to arrays and add to FAISS with ids
    if to_add_vecs:
        vecs = np.vstack(to_add_vecs).astype("float32")
        ids = np.array(to_add_ids, dtype="int64")
        try:
            index.add_with_ids(vecs, ids)
        except Exception as e:
            # If add_with_ids fails (e.g., collisions), rebuild index instead
            print("faiss add_with_ids failed:", e, "-> rebuilding index from DB")
            rebuild_index()
            return

    faiss.write_index(index, INDEX_PATH)

def query_index(query: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Query FAISS index and return (chunk_id, distance, row-metadata-dict)."""
    init_db()
    index = _init_faiss()

    q_emb = embed_texts([query]).astype("float32")
    D, I = index.search(q_emb, top_k)

    results = []
    with _get_conn() as conn:
        for faiss_id, dist in zip(I[0], D[0]):
            if faiss_id == -1:
                continue
            # find chunk_id from mapping
            cur = conn.execute("SELECT chunk_id FROM faiss_map WHERE faiss_id = ?", (int(faiss_id),))
            r = cur.fetchone()
            if not r:
                continue
            chunk_id = r[0]
            cur = conn.execute("SELECT chunk_id, doc_id, page_number, chunk_type, content, metadata FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cur.fetchone()
            if not row:
                continue
            cid, doc_id, page, ctype, content, metadata = row
            # metadata stored as JSON
            try:
                meta_obj = json.loads(metadata) if metadata else {}
            except Exception:
                meta_obj = {}
            # ensure metadata dict contains doc_id/page_number for caller convenience
            meta_obj.setdefault("doc_id", doc_id)
            meta_obj.setdefault("page_number", page)
            results.append((cid, float(dist), {
                "doc_id": doc_id,
                "page_number": page,
                "chunk_type": ctype,
                "content": content,
                "metadata": meta_obj
            }))
    return results

def rebuild_index(batch_size: int = 128):
    """
    Rebuild FAISS index from DB in batches. Also rebuilds faiss_map mapping.
    """
    print("⚡ Rebuilding ANN index from scratch...")
    init_db()
    clear_index()  # clears faiss_map rows and deletes index file
    index = _init_faiss()

    chunks = db_utils.get_all_chunks()
    print(f"📦 Found {len(chunks)} chunks in DB to re-index")

    if not chunks:
        print("⚠️ No chunks found in DB.")
        return

    with _get_conn() as conn:
        # Reset faiss_map table
        conn.execute("DELETE FROM faiss_map")
        conn.commit()

        next_id = 1
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [c["content"] for c in batch]
            embeddings = embed_texts(texts).astype("float32")

            ids = []
            for c in batch:
                cid = str(c["chunk_id"])
                fid = next_id
                next_id += 1
                ids.append(fid)
                conn.execute("INSERT OR REPLACE INTO faiss_map(faiss_id, chunk_id) VALUES (?, ?)", (fid, cid))
            conn.commit()

            ids_arr = np.array(ids, dtype="int64")
            vecs = np.vstack([e for e in embeddings]).astype("float32")
            index.add_with_ids(vecs, ids_arr)

    faiss.write_index(index, INDEX_PATH)
    print("✅ ANN index rebuilt successfully.")