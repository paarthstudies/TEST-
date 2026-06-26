"""
config.py – Central configuration for the RAG backend.

All tuneable constants live here so nothing is hard-coded in business logic.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root directory for the LanceDB database (created automatically at startup)
LANCEDB_URI: str = "./lancedb_store"

# LanceDB table that holds all ingested chunks
TABLE_NAME: str = "rag_chunks"

# Temporary upload directory (cleaned up after ingestion)
UPLOAD_DIR: Path = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embedding model (runs 100 % locally via Hugging Face sentence-transformers)
# ---------------------------------------------------------------------------

# Small but highly accurate general-purpose embedding model from BAAI
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

# Dimensionality produced by bge-small-en-v1.5
EMBEDDING_DIM: int = 384

# ---------------------------------------------------------------------------
# Ollama (local LLM daemon)
# ---------------------------------------------------------------------------

# The Ollama model tag to use for generation.
# Run `ollama pull llama3` or `ollama pull mistral` beforehand.
OLLAMA_MODEL: str = "llama3"

# Base URL for the Ollama REST API (default when running locally)
OLLAMA_HOST: str = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Retrieval / hybrid search
# ---------------------------------------------------------------------------

# Number of chunks to retrieve per query
TOP_K: int = 6

# LanceDB hybrid search alpha values:
#   alpha=1.0  → pure dense (vector) search
#   alpha=0.0  → pure BM25  (full-text) search
#
# "standard"  documents: 60 % dense  / 40 % BM25  → alpha = 0.6
# "tabular"   documents: 30 % dense  / 70 % BM25  → alpha = 0.3
#   (Tabular rows have highly structured, keyword-rich serialisations, so BM25
#    scores matter more for exact entity / value look-ups.)
HYBRID_ALPHA_STANDARD: float = 0.6
HYBRID_ALPHA_TABULAR: float = 0.3

# ---------------------------------------------------------------------------
# Text splitting (standard documents only)
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 512       # characters per chunk
CHUNK_OVERLAP: int = 64     # character overlap between consecutive chunks
