# scripts/fix_chunks_uuid.py
import sqlite3
import os

DB_PATH = os.path.join(os.getcwd(), "data", "app.db")

with sqlite3.connect(DB_PATH) as conn:
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("""
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            page_number INTEGER,
            chunk_type TEXT,
            content TEXT,
            metadata TEXT
        )
    """)
    conn.commit()
    print("Chunks table dropped and recreated with chunk_id as TEXT")
    print("Please run scripts/pop_chunks.py to repopulate the chunks table.")