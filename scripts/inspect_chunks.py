# scripts/inspect_chunks.py
from db import utils

chunks = utils.get_all_chunks()
print(f"Total chunks: {len(chunks)}\n")

for c in chunks[:5]:
    print(c["chunk_id"], "|", c["chunk_type"], "|", c["content"][:80], "...")
