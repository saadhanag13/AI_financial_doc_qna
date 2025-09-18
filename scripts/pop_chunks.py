# scripts/pop_chunks.py
import os
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import processor, chunker, embedder
from db import utils

if __name__ == "__main__":
    utils.init_db()

    uploads_dir = os.path.join("data", "uploads")
    pdf_files = glob.glob(os.path.join(uploads_dir, "*.pdf"))

    total_chunks = 0
    for pdf in pdf_files:
        print(f"📂 Processing {pdf}")
        doc = processor.process_file(pdf)
        chunks = chunker.chunk_document(doc)
        print(f"✅ Created {len(chunks)} chunks from {os.path.basename(pdf)}")

        embedder.add_chunks(chunks)
        total_chunks += len(chunks)

    print(f"🎉 Finished populating DB with {total_chunks} chunks!")
