"""
embedder.py – Singleton wrapper around the Hugging Face sentence-transformer.

Uses BAAI/bge-small-en-v1.5 which works entirely offline once downloaded.
The model is loaded once at process startup and reused for every request,
keeping memory usage flat and eliminating per-request load latency.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.  lru_cache(maxsize=1) ensures the
    model is instantiated exactly once per process regardless of how many
    concurrent requests arrive.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded successfully.")
    return model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of strings into their dense vector embeddings.

    Args:
        texts: Raw text strings to embed.

    Returns:
        A list of float lists (one per input text), ready for LanceDB ingestion.
    """
    model = _load_model()
    # normalize_embeddings=True is recommended by BAAI for cosine-similarity search
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    # Convert numpy array to plain Python lists for JSON / Arrow serialisation
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.  BGE models expect the query to be prefixed
    with "Represent this sentence: " for retrieval tasks, but the instruction
    variant (bge-small-en-v1.5) handles this internally.

    Args:
        query: The user's natural-language question.

    Returns:
        A float list representing the dense query vector.
    """
    return embed_texts([query])[0]
