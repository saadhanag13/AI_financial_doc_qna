# scripts/fix_chunks_table.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import utils

import sqlite3

DB_PATH = os.path.join("data", "app.db")

with sqlite3.connect(DB_PATH) as conn:
    cur = conn.cursor()
    
    # Drop the old table
    cur.execute("DROP TABLE IF EXISTS chunks;")
    print("Old chunks table dropped.")
    
    # Recreate table with chunk_id as TEXT
    cur.execute("""
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
    print("Chunks table recreated successfully.")
