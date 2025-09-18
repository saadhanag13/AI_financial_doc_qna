# backend/__init__.py

# Import all main modules to make them accessible via `from backend import ...`
from . import processor
from . import chunker
from . import embedder
from . import qa

# Optional: define what is exported when using `from backend import *`
__all__ = ["processor", "chunker", "embedder", "qa"]
