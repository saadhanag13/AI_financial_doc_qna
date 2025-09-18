# backend/debug_retriever.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend import retriever

if __name__ == "__main__":
    query = "What is the net income?"
    print(f"🔎 Testing query: {query}\n")

    chunks = retriever.retrieve(query, top_k=5, use_reranking=True, filter_redundant=False)
    print(f"Retrieved {len(chunks)} chunks:\n")

    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("final_score", chunk.get("score", 0))
        print(f"{i}. Score={score:.3f} | Type={chunk['chunk_type']} | Source={chunk['metadata']['source']}")
        print(f"   Content: {chunk['content'][:200]}...\n")
