import sqlite3

# Connect to your database
conn = sqlite3.connect("data/app.db")

# Get table info for 'chunks'
cur = conn.execute("PRAGMA table_info(chunks);")
columns = cur.fetchall()

# Print column info
for col in columns:
    print(col)

conn.close()
