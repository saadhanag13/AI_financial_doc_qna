#scripts/rebuild_index.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend import embedder
from db import utils 

if __name__ == "__main__":
    print("Rebuilding ANN index...")
    utils.init_db()
    embedder.rebuild_index()
    print("✅ Index rebuilt successfully")
