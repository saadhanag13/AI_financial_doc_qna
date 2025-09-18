# inspect_db.py
import sqlite3

DB = "data/app.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

# list tables
tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", tables)

# check chunk count
count = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
print("Chunks count:", count)

conn.close()
