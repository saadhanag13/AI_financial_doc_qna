#models/config.py

import os

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM model (handled via Ollama)
LLM_MODEL = os.getenv("MODEL_NAME", "llama3.1")

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))