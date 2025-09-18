# db/utils.py

# import os
# import sqlite3
# import datetime
# from typing import List, Dict, Any
# import json

# DB_PATH = os.path.join(os.getcwd(), "data", "app.db")
# os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# def _get_conn():
#     conn = sqlite3.connect(DB_PATH)
#     conn.execute("PRAGMA journal_mode=WAL;")
#     return conn


# def init_db():
#     with _get_conn() as conn:
#         conn.execute("""
#         CREATE TABLE IF NOT EXISTS documents (
#             doc_id TEXT PRIMARY KEY,
#             filename TEXT,
#             path TEXT,
#             uploaded_at TEXT
#         )
#         """)
#         conn.execute("""
#         CREATE TABLE IF NOT EXISTS conversations (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             session_id TEXT,
#             query TEXT,
#             response TEXT,
#             timestamp TEXT
#         )
#         """)
#         conn.commit()


# def init_chunks_db():
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


# def add_document(doc_id: str, filename: str, path: str):
#     with _get_conn() as conn:
#         conn.execute(
#             "INSERT OR REPLACE INTO documents(doc_id, filename, path, uploaded_at) VALUES (?, ?, ?, ?)",
#             (doc_id, filename, path, datetime.datetime.utcnow().isoformat())
#         )
#         conn.commit()


# def list_documents() -> List[Dict[str, Any]]:
#     with _get_conn() as conn:
#         cur = conn.execute("SELECT doc_id, filename, uploaded_at FROM documents ORDER BY uploaded_at DESC")
#         rows = cur.fetchall()
#     return [{"doc_id": r[0], "filename": r[1], "uploaded_at": r[2]} for r in rows]


# def add_conversation(session_id: str, query: str, response: str):
#     with _get_conn() as conn:
#         conn.execute(
#             "INSERT INTO conversations(session_id, query, response, timestamp) VALUES (?, ?, ?, ?)",
#             (session_id, query, response, datetime.datetime.utcnow().isoformat())
#         )
#         conn.commit()


# def get_conversation(session_id: str) -> List[Dict[str, Any]]:
#     with _get_conn() as conn:
#         cur = conn.execute(
#             "SELECT query, response, timestamp FROM conversations WHERE session_id=? ORDER BY id ASC",
#             (session_id,)
#         )
#         rows = cur.fetchall()
#     return [{"query": r[0], "response": r[1], "timestamp": r[2]} for r in rows]


# def get_all_chunks() -> List[Dict[str, Any]]:
#     """Fetch all chunks stored in the DB (used for rebuild_index)."""
#     with _get_conn() as conn:
#         cur = conn.execute(
#             "SELECT chunk_id, doc_id, page_number, chunk_type, content, metadata FROM chunks ORDER BY rowid"
#         )
#         rows = cur.fetchall()

#     chunks = []
#     for r in rows:
#         chunk_id, doc_id, page_number, chunk_type, content, metadata = r
#         try:
#             metadata = json.loads(metadata) if metadata else {}
#         except Exception:
#             metadata = {}
#         chunks.append({
#             "chunk_id": chunk_id,
#             "doc_id": doc_id,
#             "page_number": page_number,
#             "chunk_type": chunk_type,
#             "content": content,
#             "metadata": metadata
#         })
#     return chunks


# if __name__ == "__main__":
#     init_db()
#     init_chunks_db()
#     add_document("demo1", "file.pdf", "/path/to/file.pdf")
#     print(list_documents())
#     add_conversation("sess1", "What is revenue?", "Revenue is 123")
#     print(get_conversation("sess1"))

#     # Debug: fetch chunks
#     print("Chunks in DB:", get_all_chunks())

import os
import sqlite3
import datetime
from typing import List, Dict, Any
import json

DB_PATH = os.path.join(os.getcwd(), "data", "app.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    with _get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT,
            path TEXT,
            uploaded_at TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            query TEXT,
            response TEXT,
            timestamp TEXT
        )
        """)
        # chunks table (if missing) create with TEXT chunk_id
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
        # mapping from faiss_id -> chunk_id
        conn.execute("""
        CREATE TABLE IF NOT EXISTS faiss_map (
            faiss_id INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE
        )
        """)
        conn.commit()

def get_all_chunks() -> List[Dict[str, Any]]:
    """
    Return all chunks as a list of dicts:
    [{chunk_id, doc_id, page_number, chunk_type, content, metadata(dict)}]
    """
    with _get_conn() as conn:
        cur = conn.execute("SELECT chunk_id, doc_id, page_number, chunk_type, content, metadata FROM chunks")
        rows = cur.fetchall()
    out = []
    for r in rows:
        chunk_id, doc_id, page, ctype, content, metadata = r
        try:
            meta_obj = json.loads(metadata) if metadata else {}
        except Exception:
            try:
                # fallback: try python literal eval
                import ast
                meta_obj = ast.literal_eval(metadata)
            except Exception:
                meta_obj = {}
        out.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "page_number": page,
            "chunk_type": ctype,
            "content": content,
            "metadata": meta_obj
        })
    return out

# existing helper functions left as before
def add_document(doc_id: str, filename: str, path: str):
    with _get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO documents(doc_id, filename, path, uploaded_at) VALUES (?, ?, ?, ?)",
                     (doc_id, filename, path, datetime.datetime.utcnow().isoformat()))
        conn.commit()

def list_documents() -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        cur = conn.execute("SELECT doc_id, filename, uploaded_at FROM documents ORDER BY uploaded_at DESC")
        rows = cur.fetchall()
    return [{"doc_id": r[0], "filename": r[1], "uploaded_at": r[2]} for r in rows]

def add_conversation(session_id: str, query: str, response: str):
    with _get_conn() as conn:
        conn.execute("INSERT INTO conversations(session_id, query, response, timestamp) VALUES (?, ?, ?, ?)",
                     (session_id, query, response, datetime.datetime.utcnow().isoformat()))
        conn.commit()

def get_conversation(session_id: str) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        cur = conn.execute("SELECT query, response, timestamp FROM conversations WHERE session_id=? ORDER BY id ASC", (session_id,))
        rows = cur.fetchall()
    return [{"query": r[0], "response": r[1], "timestamp": r[2]} for r in rows]

if __name__ == "__main__":
    init_db()