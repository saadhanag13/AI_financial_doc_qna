import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend import processor, chunker

pdf_path = os.path.join("Sample-Financial-Statements-1.pdf")  # adjust if in data/uploads

# Wrap path in a file-like object
with open(pdf_path, "rb") as f:
    doc = processor.process_file(f)

print(f"✅ Processed file: {doc['filename']} with {len(doc['pages'])} pages")

# Chunk the doc
chunks = chunker.chunk_document(doc)
print(f"✅ Created {len(chunks)} chunks")

# Show a preview of chunks that might contain 'net income'
for c in chunks:
    if "net income" in c["content"].lower():
        print("\n🔎 Found potential match in chunk:")
        print(json.dumps(c, indent=2, ensure_ascii=False))