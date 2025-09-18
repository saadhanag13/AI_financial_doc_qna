#db/utils.py

import os
import sqlite3
import datetime
from typing import List, Dict, Any

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
        conn.commit()

def init_chunks_db():
    with _get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            page_number INTEGER,
            chunk_type TEXT,
            content TEXT,
            metadata TEXT
        )
        """)
        conn.commit()



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
        conn.execute("INSERT INTO conversations(session_id, query, response, timestamp) VALUES (?, ?, ?, ?)", (session_id, query, response, datetime.datetime.utcnow().isoformat()))
        conn.commit()


def get_conversation(session_id: str) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        cur = conn.execute("SELECT query, response, timestamp FROM conversations WHERE session_id=? ORDER BY id ASC", (session_id,))
        rows = cur.fetchall()
    return [{"query": r[0], "response": r[1], "timestamp": r[2]} for r in rows]

if __name__ == "__main__":
    init_db()
    init_chunks_db()
    add_document("demo1", "file.pdf", "/path/to/file.pdf")
    print(list_documents())
    add_conversation("sess1", "What is revenue?", "Revenue is 123")
    print(get_conversation("sess1"))
