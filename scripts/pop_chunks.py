#scripts/pop_chunks.py

import sqlite3
import os
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import utils

utils.init_db()
utils.init_chunks_db()

DB_PATH = os.path.join(os.getcwd(), "data", "app.db")

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def chunk_text(text, chunk_size=200):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def populate_chunks():
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("SELECT doc_id, filename FROM documents")
    documents = cur.fetchall()

    chunk_id = 1
    for doc_id, filename in documents:
        
        text = f"Contents of {filename}"

        for i, chunk in enumerate(chunk_text(text), start=1):
            cur.execute("""
                INSERT INTO chunks (chunk_id, doc_id, page_number, chunk_type, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk_id, doc_id, i, "text", chunk, "{}"))
            chunk_id += 1

    conn.commit()
    print("Chunks populated successfully!")

if __name__ == "__main__":
    populate_chunks()
