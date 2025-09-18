#models/embeddings.py

import threading
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from models import config

_model = None
_lock = threading.Lock()


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    return np.array(model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True))
