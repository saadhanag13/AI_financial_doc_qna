# tests/pipeline.py

import os
import json
import uuid
import pytest

from backend import processor, chunker, embedder, retriever
from db import db_utils

SAMPLE_TEXT = """Revenue for 2023 was 12345 INR. Net profit stood at 6789 INR."""


def test_chunker_text():
    doc = {
        "doc_id": "demo",
        "filename": "demo.txt",
        "pages": [{"page_number": 1, "text": SAMPLE_TEXT, "tables": []}]
    }
    chunks = chunker.chunk_document(doc, max_words=50, overlap=10)
    assert len(chunks) >= 1
    assert any("Revenue" in c["content"] for c in chunks)


def test_embedder_and_retriever(tmp_path):
    db_utils.init_db()
    doc_id = uuid.uuid4().hex
    doc = {
        "doc_id": doc_id,
        "filename": "demo.txt",
        "pages": [{"page_number": 1, "text": SAMPLE_TEXT, "tables": []}]
    }
    chunks = chunker.chunk_document(doc)
    embedder.add_chunks(chunks)

    results = retriever.retrieve("What was the revenue in 2023?", top_k=3)
    assert results, "Retriever returned no results"
    assert any("Revenue" in r["content"] for r in results)


def test_processor_textfile(tmp_path):
    file_path = tmp_path / "demo.txt"
    file_path.write_text(SAMPLE_TEXT)

    doc_json = processor.process_file(str(file_path))
    assert "pages" in doc_json
    assert doc_json["pages"][0]["text"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
