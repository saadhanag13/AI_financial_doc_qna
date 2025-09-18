#scripts/migrate_chunks.py 

import sqlite3, json, ast, re, os, sys

DB_PATH = os.path.join(os.getcwd(), "data", "app.db")

def sanitize_content(s: str) -> str:
    if s is None:
        return ""
    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="replace")
    else:
        s = str(s)

    s = re.sub(r"[^\x09\x0A\x20-\x7E\u0080-\uFFFF]", " ", s)
    # collapse multiple whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_metadata(raw):
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    txt = str(raw).strip()
    if not txt:
        return {}

    try:
        return json.loads(txt)
    except Exception:
        pass

    try:
        return ast.literal_eval(txt)
    except Exception:
        pass

    try:
        candidate = re.sub(r"'", '"', txt)
        candidate = re.sub(r",\s*}", "}", candidate)
        return json.loads(candidate)
    except Exception:
        return {}

def migrate():
    if not os.path.exists(DB_PATH):
        print("DB not found at", DB_PATH); sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
    if not cur.fetchone():
        print("No existing 'chunks' table found — nothing to migrate.")
        conn.close()
        return

    print("Reading old chunks...")
    cur.execute("SELECT chunk_id, doc_id, page_number, chunk_type, content, metadata FROM chunks")
    rows = cur.fetchall()
    print(f"Found {len(rows)} rows in old chunks table.")

    # Create new chunks table
    cur.execute("PRAGMA foreign_keys = OFF;")
    conn.commit()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks_new (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT,
        page_number INTEGER,
        chunk_type TEXT,
        content TEXT,
        metadata TEXT
    )
    """)
    conn.commit()

    inserted = 0
    for r in rows:
        old_chunk_id, doc_id, page_number, chunk_type, content, metadata = r
        chunk_id = str(old_chunk_id) if old_chunk_id is not None else None
        if chunk_id is None or chunk_id == "":
            chunk_id = f"old-{inserted+1}"

        content_clean = sanitize_content(content)
        meta_obj = normalize_metadata(metadata)
        meta_json = json.dumps(meta_obj, ensure_ascii=False)

        try:
            cur.execute("""
            INSERT OR REPLACE INTO chunks_new (chunk_id, doc_id, page_number, chunk_type, content, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk_id, str(doc_id) if doc_id is not None else "", page_number if page_number is not None else 0,
                  str(chunk_type) if chunk_type is not None else "text", content_clean, meta_json))
            inserted += 1
        except Exception as e:
            print("Failed inserting row:", e)

    conn.commit()
    print(f"Inserted {inserted} rows into chunks_new.")

    # Backup old table name
    cur.execute("ALTER TABLE chunks RENAME TO chunks_old")
    conn.commit()

    # Rename new -> chunks
    cur.execute("ALTER TABLE chunks_new RENAME TO chunks")
    conn.commit()

    print("Migration complete. Old table renamed to 'chunks_old'.")
    conn.close()

if __name__ == "__main__":
    migrate()